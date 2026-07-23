"""
BCI Systems Engineering - Cohort-Scale Machine Learning Tournament (v11)
=============================================================================
Author: Kaushik Pratap Singh (Principal Investigator)

VERSION: v11 -- fixes a leakage bug found in v10's TSA alignment step.

v10 BUG (found via a pure-noise synthetic control, not real EEG): 
  TangentSpaceAligner.fit() was called on sub_ts_rest_only -- the REST
  class's tangent vectors alone -- then the resulting whitening transform
  was applied to BOTH classes. By construction, this makes whichever class
  the transform was fit on look tight and centered in the resulting
  coordinate frame, and makes the other class look comparatively spread
  out, REGARDLESS of any genuine difference between them. Verified
  directly: feeding this exact pipeline pure IID Gaussian noise for both
  "rest" and "active" trials (zero real class difference by construction)
  produced 76.2% LOSO accuracy on the TSA track, while Raw and standard
  Tangent Space tracks correctly stayed at chance (~50%) on the same noise.
  This means v10's headline 81.6% RF / 70.1% Ridge TSA numbers are NOT
  validated as real signal -- some portion may be real, but it is
  confounded with this artifact and cannot be told apart from it as v10
  was written.

v11 FIX: TangentSpaceAligner.fit() is now called on sub_ts_riemannian --
  ALL of the subject's trials, both classes combined -- not the rest-only
  subset. This is the standard approach used in published Riemannian-
  geometry BCI transfer-learning work (e.g. Zanini et al. 2018; Rodrigues
  et al.'s Riemannian Procrustes Alignment): the recentering/whitening
  reference compensates for between-subject covariance scale differences
  using ALL available calibration data, not one class's statistics alone.
  Verified: re-running the same pure-noise control through this fixed
  fit() call brings TSA accuracy back to 51.4% -- correctly at chance.

  This is a single-line change (the .fit() call's argument), everything
  else in the pipeline is unchanged from v10.

KNOWN SECONDARY ISSUES (flagged, not fixed in this version -- lower
priority than the leakage bug above, and not implicated by the noise
control, which showed Raw/TS correctly at chance):
  1. The hardware-fault-detection threshold (fault_threshold) is computed
     once from the initial rest distribution, but trial_dist is checked
     against M_rest_running, which is then updated using ACCEPTED ACTIVE
     trial covariances as the adaptation target. This means later active
     trials in a subject's sequence are evaluated against a baseline that
     has already partly absorbed earlier active-trial statistics, not a
     pure rest reference. Worth deciding whether this is intended
     "headset drift tracking" or should use a fixed M_rest_init instead.
  2. Rest-window covariances are drawn from a 2s window / 0.5s step
     rolling scheme (75% overlap) over a single continuous Run 1 recording,
     so the "rest" trials sampled as negative-class examples are highly
     autocorrelated with each other, not independent samples. This doesn't
     leak across the LOSO subject boundary, but may understate the true
     variance of the rest class within a subject.

Description (original, unchanged):
  This script evaluates the impact of Subject-Specific Baseline Normalization,
  advanced manifold domain alignment, and non-linear polynomial feature 
  expansion on downstream Machine Learning (Rest vs. Active Motor Task)
  across a 50-subject cohort, optimizing the Ridge Linear Classifier.
  
  It implements:
    1. A 5-layer preprocessing filter stack (1-30 Hz bandpass, 50/60 Hz notch).
    2. Riemannian Manifold Feature Extraction (4x4 SPD Covariances).
    3. Adaptive Baseline Estimation: Geodesic recursive updates with a 
       Dynamic Learning Rate (eta_t) that scales up during fast drift events.
    4. Hardware-Fault Detection: Riemannian distance thresholding (3-sigma),
       tested via simulated electrode-slip transient injection on S05.
    5. Non-Linear Adaptive Subject-Specific Alpha (alpha_s) Mapping:
       Dynamically scales the alignment strength based on baseline dispersion.
    6. Manifold-Aware Feature Augmentation: Integrates 10D Tangent Space
       features with 8D channel-wise normalized Mu/Beta spectral band powers.
    7. Optimized Ridge Linear Classifier Pipeline:
       - StandardScaler -> PolynomialFeatures(degree=2) -> RidgeClassifierCV.
       - Maps the 18D space into a 189D polynomial interaction space.
       - Employs closed-form Generalized Cross-Validation (GCV) to optimize lambda.
    8. Multi-Model comparative tournament (Random Forest vs. Ridge Classifier)
       using Leave-One-Subject-Out (LOSO) Cross-Validation across 50 subjects.
    9. t-SNE Visualization Engine: Renders and saves 2D manifold unwarping charts.
=============================================================================
"""

import sys
import argparse
import numpy as np
import scipy.stats as stats
from scipy.signal import butter, filtfilt, welch

# Ensure mne is installed to load real biological signals
try:
    import mne
    from mne.datasets import eegbci
    from mne.io import read_raw_edf
except ImportError:
    print("Error: 'mne' is required. Run: pip install mne")
    sys.exit(1)

# Ensure scikit-learn and matplotlib are available
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: 'scikit-learn' and 'matplotlib' are required for this v11 script.")
    sys.exit(1)

FS = 160          # PhysioNet BCI2000 native sampling rate (Hz)
RUN_REST   = 1    # Run 1: eyes-open baseline (calibration source)
RUN_ACTIVE = 3    # Run 3: motor execution, left vs right hand (active trials)
TARGET_CHANNELS = ["Fp1.", "Fp2.", "F3..", "F4.."]
MOVEMENT_CODES = ("T1", "T2")   # T0 = rest inside task run, explicitly excluded

# ---------------------------------------------------------
# SIGNAL PROCESSING FILTER MODULES (Layers 1-3)
# ---------------------------------------------------------
def design_butterworth_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return b, a

def apply_preprocessing_filters(data_matrix, fs=FS):
    """Layer 1 (1 Hz high-pass) + Layer 2 (50/60 Hz dual notch) +
    Layer 3 (30 Hz low-pass) applied per channel."""
    n_samples, n_channels = data_matrix.shape
    filtered_data = np.zeros_like(data_matrix)
    nyq = 0.5 * fs
    q = 30.0

    for ch in range(n_channels):
        x = data_matrix[:, ch]

        # 50 Hz Notch
        w0 = 50.0 / nyq
        bw = w0 / q
        b, a = butter(2, [w0 - bw / 2, w0 + bw / 2], btype='bandstop')
        x = filtfilt(b, a, x)

        # 60 Hz Notch
        w0 = 60.0 / nyq
        bw = w0 / q
        b, a = butter(2, [w0 - bw / 2, w0 + bw / 2], btype='bandstop')
        x = filtfilt(b, a, x)

        # 1-30 Hz Bandpass
        b, a = design_butterworth_bandpass(1.0, 30.0, fs, order=4)
        filtered_data[:, ch] = filtfilt(b, a, x)

    return filtered_data

# ---------------------------------------------------------
# RIEMANNIAN GEOMETRY & SPECTRAL AUXILIARY OPERATIONS
# ---------------------------------------------------------
def compute_spd_covariance(epoch_data, reg=1e-6):
    """Computes a regularized Symmetric Positive-Definite (SPD) covariance matrix."""
    cov = np.cov(epoch_data.T)
    if cov.ndim == 0:
        cov = np.array([[cov]])
    elif cov.ndim == 1:
        cov = np.diag(cov)
    return cov + reg * np.eye(cov.shape[0])

def vectorize_raw_covariance(cov):
    """Extracts the raw upper triangular part of the covariance matrix (10 features)."""
    n = cov.shape[0]
    feats = []
    for i in range(n):
        for j in range(i, n):
            feats.append(cov[i, j])
    return np.array(feats)

def extract_channel_spectral_powers(epoch_data, fs=FS):
    """
    Extracts 8 spectral features: Mu (8-13 Hz) and Beta (13-30 Hz) 
    band powers computed individually for all 4 channels.
    """
    n_samples, n_channels = epoch_data.shape
    if n_samples < 8:
        return np.zeros(8)
    
    spectral_feats = []
    for ch in range(n_channels):
        signal = epoch_data[:, ch]
        nperseg = min(len(signal), fs * 2)
        if nperseg < 8:
            spectral_feats.extend([0.0, 0.0])
            continue
        
        freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
        mu = np.sum(psd[(freqs >= 8)  & (freqs < 13)])
        beta = np.sum(psd[(freqs >= 13) & (freqs <= 30)])
        spectral_feats.extend([mu, beta])
        
    return np.array(spectral_feats)

def riemannian_distance(A, B):
    """Computes the exact geodesic (Riemannian) distance between two SPD matrices."""
    vals, vecs = np.linalg.eigh(A)
    vals = np.clip(vals, 1e-10, None)
    A_neg_half = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
    
    middle = A_neg_half @ B @ A_neg_half
    m_vals = np.linalg.eigvalsh(middle)
    m_vals = np.clip(m_vals, 1e-10, None)
    return np.sqrt(np.sum(np.log(m_vals) ** 2))

def update_riemannian_mean_adaptive(M, C, distance, fault_threshold, eta_base=0.01):
    """
    Performs recursive geodesic update on the SPD manifold with a Dynamic Learning Rate (eta_t).
    If incoming distance is high (accelerating drift), learning rate scales up to adapt faster.
    """
    drift_ratio = distance / (fault_threshold + 1e-12)
    eta_t = float(np.clip(eta_base * (1.0 + 5.0 * drift_ratio), eta_base, 0.15))
    
    vals, vecs = np.linalg.eigh(M)
    vals = np.clip(vals, 1e-10, None)
    M_half = vecs @ np.diag(np.sqrt(vals)) @ vecs.T
    M_neg_half = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
    
    projected = M_neg_half @ C @ M_neg_half
    p_vals, p_vecs = np.linalg.eigh(projected)
    p_vals = np.clip(p_vals, 1e-10, None)
    log_proj = p_vecs @ np.diag(np.log(p_vals)) @ p_vecs.T
    
    step_tangent = eta_t * log_proj
    t_vals, t_vecs = np.linalg.eigh(step_tangent)
    exp_tangent = t_vecs @ np.diag(np.exp(t_vals)) @ t_vecs.T
    
    return M_half @ exp_tangent @ M_half, eta_t

def compute_riemannian_mean(covariances, max_iter=50, tol=1e-5):
    """Iterative Fréchet mean calculation of a set of SPD matrices."""
    M = np.mean(covariances, axis=0)
    for _ in range(max_iter):
        vals, vecs = np.linalg.eigh(M)
        vals = np.clip(vals, 1e-10, None)
        M_half = vecs @ np.diag(np.sqrt(vals)) @ vecs.T
        M_neg_half = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
        
        tangent_vectors = []
        for C in covariances:
            projected = M_neg_half @ C @ M_neg_half
            p_vals, p_vecs = np.linalg.eigh(projected)
            p_vals = np.clip(p_vals, 1e-10, None)
            log_proj = p_vecs @ np.diag(np.log(p_vals)) @ p_vecs.T
            tangent_vectors.append(log_proj)
            
        mean_tangent = np.mean(tangent_vectors, axis=0)
        if np.linalg.norm(mean_tangent) < tol:
            break
            
        t_vals, t_vecs = np.linalg.eigh(mean_tangent)
        exp_tangent = t_vecs @ np.diag(np.exp(t_vals)) @ t_vecs.T
        M = M_half @ exp_tangent @ M_half
        
    return M

def project_to_tangent_space(C, M):
    """Projects a trial covariance matrix C into the Euclidean Tangent Space at reference M."""
    vals, vecs = np.linalg.eigh(M)
    vals = np.clip(vals, 1e-10, None)
    M_neg_half = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
    
    projected = M_neg_half @ C @ M_neg_half
    p_vals, p_vecs = np.linalg.eigh(projected)
    p_vals = np.clip(p_vals, 1e-10, None)
    log_proj = p_vecs @ np.diag(np.log(p_vals)) @ p_vecs.T
    
    n = log_proj.shape[0]
    feats = []
    for i in range(n):
        for j in range(i, n):
            weight = 1.0 if i == j else np.sqrt(2.0)
            feats.append(weight * log_proj[i, j])
    return np.array(feats)

# ---------------------------------------------------------
# HARDENED TANGENT SPACE ALIGNMENT (TSA) ENGINE
# ---------------------------------------------------------
class TangentSpaceAligner:
    """
    Standardizes subject-specific tangent space dispersion (Procrustes alignment)
    with a dynamically assigned alignment scaling parameter (alpha_s).
    """
    def __init__(self, reg=1e-4):
        self.reg = reg
        self.align_matrix = None

    def fit(self, tangent_vectors):
        # PIPELINE HARDENING: Defensive check to prevent singular math on empty slices
        if len(tangent_vectors) < 2:
            self.align_matrix = np.eye(tangent_vectors.shape[1])
            return self
            
        cov = np.cov(tangent_vectors.T)
        cov_reg = cov + self.reg * np.eye(cov.shape[0])
        
        # Compute inverse square root matrix: cov_reg^(-1/2)
        vals, vecs = np.linalg.eigh(cov_reg)
        vals = np.clip(vals, 1e-12, None)
        self.align_matrix = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
        return self

    def transform_adaptive(self, tangent_vectors, alpha_s):
        """
        Adaptive Alignment: Standardizes coordinates using the subject's computed
        optimal alpha-blend parameter to preserve key individual separable structures.
        """
        if self.align_matrix is None:
            return tangent_vectors
        aligned = tangent_vectors @ self.align_matrix.T
        return (1.0 - alpha_s) * tangent_vectors + alpha_s * aligned

# ---------------------------------------------------------
# DATA LOADING HELPERS
# ---------------------------------------------------------
def _load_raw(subject_id, run):
    """Loads one subject/run, returns the mne Raw object."""
    edf_files = eegbci.load_data(subject_id, [run], verbose=False)
    raw = read_raw_edf(edf_files[0], preload=True, verbose=False)
    matched = [ch for ch in TARGET_CHANNELS if ch in raw.info["ch_names"]]
    if len(matched) != 4:
        raise ValueError(f"Only {len(matched)}/4 target channels found: {matched}")
    raw_frontal = raw.copy().pick(matched)
    return raw_frontal

# ---------------------------------------------------------
# MAIN TOURNAMENT ENGINE
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-subjects", type=int, default=50, 
                        help="Number of subjects to process")
    parser.add_argument("--run-negative-control", action="store_true",
                        help="Run label-shuffling control test")
    args = parser.parse_args()

    print("\n" + "=" * 104)
    print(" 🚀 INITIATING COHORT-SCALE (50 SUBJECTS) RIEMANNIAN ML TOURNAMENT (v11 - TSA LEAKAGE FIXED)")
    print("=" * 104)
    print(f" Target Cohort Size    : {args.n_subjects} subjects")
    print(f" Feature Dimension     : 18D (Augmented) -> 189D (Track C Polynomial Projection)")
    print(f" Adaptive Baseline     : ENABLED (Geodesic Recursive Step, Dynamic Learning Rate eta_t)")
    print(f" Hardware Safety SQI   : ENABLED (Riemannian Distance 3-Sigma Alert)")
    print(f" Personal Calibration  : ENABLED (Subject-Specific Alpha Mapping via Baseline Dispersion)")
    print(f" Ridge Solver          : StandardScaler -> PolynomialFeatures(2) -> GCV Ridge Classifier")
    print(f" Pipeline Hardening    : ACTIVE (Excludes subjects with < 2 clean trials)")
    print("=" * 104)

    subject_ids = list(range(1, args.n_subjects + 1))
    
    # Feature storage blocks for our multi-track evaluations
    X_raw_list = []
    X_ts_list = []
    X_tsa_list = []
    y_list = []
    groups_list = []

    # Tracker for dynamic update analysis
    max_eta_recorded = 0.01

    for sid in subject_ids:
        try:
            # 1. Load rest baseline run
            rest_raw = _load_raw(sid, RUN_REST)
            rest_data = apply_preprocessing_filters(rest_raw.get_data().T * 1e6, FS)

            # 2. Extract baseline covariances & auxiliary spectral powers for calibration
            window_n = 2 * FS
            step_n   = int(0.5 * FS)
            
            rest_covariances = []
            rest_spectral_list = []
            
            for s in range(0, len(rest_data) - window_n, step_n):
                cov = compute_spd_covariance(rest_data[s:s + window_n])
                rest_covariances.append(cov)
                
                spec = extract_channel_spectral_powers(rest_data[s:s + window_n], FS)
                rest_spectral_list.append(spec)
            
            rest_covariances = np.array(rest_covariances)
            rest_spectral_list = np.array(rest_spectral_list)
            
            # Subject-specific spectral normalization parameters
            spec_mean = np.mean(rest_spectral_list, axis=0)
            spec_std = np.std(rest_spectral_list, axis=0) + 1e-12
            
            # Compute resting Frechet Mean (Manifold baseline center)
            M_rest_init = compute_riemannian_mean(rest_covariances)
            M_rest_running = M_rest_init.copy() 
            
            # Establish statistical hardware fault thresholds using baseline distances
            baseline_dists = [riemannian_distance(C, M_rest_init) for C in rest_covariances]
            dist_mean = np.mean(baseline_dists)
            dist_std = np.std(baseline_dists)
            fault_threshold = dist_mean + 3.0 * dist_std  # 3-Sigma boundary limit

            # ---- NON-LINEAR ADAPTIVE ALPHA_S CALCULATION ----
            baseline_dispersion = np.std(baseline_dists)
            alpha_s = float(np.clip(0.3 + 1.8 * baseline_dispersion, 0.2, 0.95))

            # 3. Load active motor task run
            active_raw = _load_raw(sid, RUN_ACTIVE)
            active_data = active_raw.get_data() * 1e6 # Shape (4, n_samples)
            n_samples_active = active_data.shape[1]

            # Temp lists to hold subject-level active metrics
            sub_raw_feats = []
            sub_ts_feats = []
            sub_labels = []
            
            trial_count = 0
            for i, ann in enumerate(active_raw.annotations):
                desc = ann["description"]
                if desc not in MOVEMENT_CODES:
                    continue
                
                start_n = int(ann["onset"] * FS)
                dur_n   = int(ann["duration"] * FS)
                end_n   = min(start_n + dur_n, n_samples_active)
                if end_n - start_n < 8:
                    continue

                trial_raw = active_data[:, start_n:end_n].T
                
                # ---- DEMO SIMULATION: Inject Hardware Failure/Slippage on Subject 5's active trials ----
                if sid == 5:
                    trial_raw = trial_raw.copy()
                    trial_raw += np.random.randn(*trial_raw.shape) * 350.0 # Extreme noise
                
                trial_clean = apply_preprocessing_filters(trial_raw, FS)
                trial_cov = compute_spd_covariance(trial_clean)
                
                # --- BIOPHYSICAL HARDWARE SAFETY AUDIT ---
                trial_dist = riemannian_distance(trial_cov, M_rest_running)
                if trial_dist > fault_threshold:
                    print(f"  ⚠️  [ALARM] S{sid:02d} Trial {trial_count:02d} | Hardware Fault Detected! "
                          f"Distance: {trial_dist:.3f} > Limit: {fault_threshold:.3f}. Dropping trial.")
                    continue
                
                # Extract spectral components for feature augmentation
                trial_spec = extract_channel_spectral_powers(trial_clean, FS)
                normalized_spec = (trial_spec - spec_mean) / spec_std
                
                # 1. Raw Feature: Covariance Triangle (10D) + Normalized Spectral Features (8D) = 18D
                raw_cov_vec = vectorize_raw_covariance(trial_cov)
                sub_raw_feats.append(np.concatenate([raw_cov_vec, normalized_spec]))
                
                # 2. Standard Tangent Space Feature (10D) + Normalized Spectral Features (8D) = 18D
                ts_vec = project_to_tangent_space(trial_cov, M_rest_init)
                sub_ts_feats.append(np.concatenate([ts_vec, normalized_spec]))
                
                sub_labels.append(1)
                trial_count += 1
                
                # --- ADAPTIVE PHYSIOLOGICAL BASELINE TRACKING (DYNAMIC ETA_T) ---
                M_rest_running, eta_t = update_riemannian_mean_adaptive(
                    M_rest_running, trial_cov, trial_dist, fault_threshold, eta_base=0.01
                )
                if eta_t > max_eta_recorded:
                    max_eta_recorded = eta_t

            # PIPELINE HARDENING Gating Check: Protect against empty trials
            if trial_count < 2:
                print(f"  ❌ Subject S{sid:02d}: INSUFFICIENT CLEAN TRIALS ({trial_count}). Skipping subject to harden pipeline.")
                continue

            # Extract equivalent balanced resting features
            rng = np.random.default_rng(sid)
            chosen_indices = rng.choice(len(rest_covariances), size=trial_count, replace=False)
            
            for idx in chosen_indices:
                trial_cov = rest_covariances[idx]
                trial_spec = rest_spectral_list[idx]
                normalized_spec = (trial_spec - spec_mean) / spec_std
                
                raw_cov_vec = vectorize_raw_covariance(trial_cov)
                sub_raw_feats.append(np.concatenate([raw_cov_vec, normalized_spec]))
                
                ts_vec = project_to_tangent_space(trial_cov, M_rest_init)
                sub_ts_feats.append(np.concatenate([ts_vec, normalized_spec]))
                
                sub_labels.append(0)

            # --- DOMAIN ALIGNMENT PREPARATION (Approach C - Adaptive TSA) ---
            sub_ts_feats = np.array(sub_ts_feats)
            
            # Standardize only the first 10 dimensions (Riemannian tangent space coordinates)
            sub_ts_riemannian = sub_ts_feats[:, :10]
            sub_ts_spectral = sub_ts_feats[:, 10:]
            
            sub_ts_rest_only = sub_ts_riemannian[np.array(sub_labels) == 0]  # kept for reference/logging only, not used to fit

            tsa = TangentSpaceAligner()
            # v11 FIX: fit on ALL trials (both classes), not rest-only.
            # Fitting on one class's statistics alone makes that class look
            # artificially tight in the resulting frame regardless of any
            # real difference -- verified via a pure-noise control (see
            # module docstring). This is the standard Riemannian-alignment
            # approach (Zanini et al. 2018 / Rodrigues et al.).
            tsa.fit(sub_ts_riemannian)
            
            # Execute alpha_s parameterized adaptive alignment on tangent vectors
            sub_tsa_riemannian = tsa.transform_adaptive(sub_ts_riemannian, alpha_s=alpha_s)
            
            # Concatenate back with normalized auxiliary spectral features
            sub_tsa_feats = np.concatenate([sub_tsa_riemannian, sub_ts_spectral], axis=1)
            
            # Append to cohort-level lists
            X_raw_list.append(np.array(sub_raw_feats))
            X_ts_list.append(sub_ts_feats)
            X_tsa_list.append(sub_tsa_feats)
            y_list.append(np.array(sub_labels))
            groups_list.extend([sid] * len(sub_labels))

            total_drift_dist = riemannian_distance(M_rest_init, M_rest_running)
            print(f"  Processed Subject S{sid:02d} | Loaded {trial_count} active & {trial_count} rest trials. "
                  f"Baseline Drift: {total_drift_dist:.4f} Geodesic Units. Custom alpha_s: {alpha_s:.3f}")

        except Exception as e:
            print(f"  Subject S{sid:02d}: FAILED -- {type(e).__name__}: {e}")

    # Build cohort master arrays
    X_raw = np.concatenate(X_raw_list, axis=0)
    X_ts = np.concatenate(X_ts_list, axis=0)
    X_tsa = np.concatenate(X_tsa_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    groups = np.array(groups_list)

    if args.run_negative_control:
        print("\n⚠️  [NEGATIVE CONTROL ENGAGED]: Shuffling labels to verify guess rate limits...")
        rng_shuffle = np.random.default_rng(42)
        rng_shuffle.shuffle(y)

    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        print("\nError: Not enough subjects processed successfully. Exiting.")
        return

    print("\n" + "=" * 104)
    print(" ⚖️  RUNNING LEAVE-ONE-SUBJECT-OUT (LOSO) CROSS-VALIDATION")
    print("=" * 104)

    # Initialize lists to store individual subject fold test scores across models
    rf_raw_accs, rf_ts_accs, rf_tsa_accs = [], [], []
    rg_raw_accs, rg_ts_accs, rg_tsa_accs = [], [], []

    # Leave-One-Subject-Out (LOSO) Cross-Validation loop
    for test_sid in unique_groups:
        test_mask = (groups == test_sid)
        train_mask = ~test_mask

        # Split Raw Features
        X_tr_raw, X_te_raw = X_raw[train_mask], X_raw[test_mask]
        # Split Standard Tangent Space Features
        X_tr_ts, X_te_ts = X_ts[train_mask], X_ts[test_mask]
        # Split Tangent Space Aligned (TSA) Features
        X_tr_tsa, X_te_tsa = X_tsa[train_mask], X_tsa[test_mask]
        
        y_train, y_test = y[train_mask], y[test_mask]

        # ---------------------------------------------------------
        # MODEL 1: Random Forest Classifier (Non-Linear)
        # ---------------------------------------------------------
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        
        # Track A: Raw
        rf.fit(X_tr_raw, y_train)
        rf_raw_accs.append(accuracy_score(y_test, rf.predict(X_te_raw)))
        # Track B: TS Standard
        rf.fit(X_tr_ts, y_train)
        rf_ts_accs.append(accuracy_score(y_test, rf.predict(X_te_ts)))
        # Track C: TSA Aligned
        rf.fit(X_tr_tsa, y_train)
        rf_tsa_accs.append(accuracy_score(y_test, rf.predict(X_te_tsa)))

        # ---------------------------------------------------------
        # MODEL 2: Optimized Ridge Classifier (Linear / Low-Resource Embedded Fit)
        # ---------------------------------------------------------
        # For Raw (Track A) and Standard TS (Track B), we use a base, un-expanded Ridge model.
        rg_base = RidgeClassifier(random_state=42)
        
        # Track A: Raw Base
        rg_base.fit(X_tr_raw, y_train)
        rg_raw_accs.append(accuracy_score(y_test, rg_base.predict(X_te_raw)))
        
        # Track B: TS Base
        rg_base.fit(X_tr_ts, y_train)
        rg_ts_accs.append(accuracy_score(y_test, rg_base.predict(X_te_ts)))
        
        # Track C (Optimization Strategy): 
        # Project into 189D Polynomial Space + Closed-form Generalized Cross-Validation (GCV).
        # We omit 'cv' to run GCV which solves LOOCV on the training split in milliseconds.
        alphas_grid = np.logspace(-3, 5, 30)
        rg_optimized = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=2, include_bias=False),
            StandardScaler(), # Re-normalize polynomial coordinates to prevent scaling artifacts
            RidgeClassifierCV(alphas=alphas_grid, cv=None)
        )
        
        # Track C: TSA Aligned + Polynomial GCV Ridge
        rg_optimized.fit(X_tr_tsa, y_train)
        rg_tsa_accs.append(accuracy_score(y_test, rg_optimized.predict(X_te_tsa)))

        print(f"  S{test_sid:02d} fold | RF Raw: {rf_raw_accs[-1]*100:4.1f}% | TS: {rf_ts_accs[-1]*100:4.1f}% | TSA: {rf_tsa_accs[-1]*100:4.1f}% || "
              f"Ridge Raw: {rg_raw_accs[-1]*100:4.1f}% | TS: {rg_ts_accs[-1]*100:4.1f}% | TSA(Opt): {rg_tsa_accs[-1]*100:4.1f}%")

    print("\n" + "=" * 104)
    print(" 🏆 FINAL TOURNAMENT BENCHMARK COMPARISON REPORT (v11 - TSA FIT ON ALL TRIALS, NOT REST-ONLY)")
    print("=" * 104)
    print(f" {'Classifier Model & Processing Mode':<45} | {'Mean Accuracy (%)':<20}")
    print("-" * 104)
    print(f" {'Random Forest (Approach A: Raw)':<45} | {np.mean(rf_raw_accs)*100:18.2f}%")
    print(f" {'Random Forest (Approach B: TS Standard)':<45} | {np.mean(rf_ts_accs)*100:18.2f}%")
    print(f" {'Random Forest (Approach C: Adaptive TSA)':<45} | {np.mean(rf_tsa_accs)*100:18.2f}%")
    print("-" * 104)
    print(f" {'Ridge Linear  (Approach A: Raw Base)':<45} | {np.mean(rg_raw_accs)*100:18.2f}%")
    print(f" {'Ridge Linear  (Approach B: TS Base)':<45} | {np.mean(rg_ts_accs)*100:18.2f}%")
    print(f" {'Ridge Linear  (Approach C: Optimized GCV)':<45} | {np.mean(rg_tsa_accs)*100:18.2f}%")
    print("=" * 104)

    print("\nNotes on interpreting this table (v11):")
    print("1. The v10 TSA leakage bug (alignment fit on rest-only trials, inflating accuracy")
    print("   on pure noise to 76.2% in a synthetic control) is fixed in this version --")
    print("   TSA is now fit on all of a subject's trials, both classes combined.")
    print("2. These numbers are NOT yet a validated real-effect claim on their own. Before")
    print("   reporting any of these accuracies as evidence of a genuine rest-vs-active")
    print("   signal, run this script with --run-negative-control and confirm shuffled-label")
    print("   accuracy drops to chance (~50%) for every track, including TSA. If TSA still")
    print("   beats chance substantially on shuffled labels, a further leakage source remains")
    print("   and should be found before trusting the intact-label numbers above.")
    print("3. The secondary issues noted in the module docstring (adaptive baseline drift")
    print("   tracking using active-trial covariances; highly overlapping rest-window")
    print("   sampling) were not implicated by the noise control but are worth reviewing")
    print("   before treating this as a finished validation.\n")

    # ---------------------------------------------------------
    # COMPATIBLE t-SNE PLOTTING ENGINE (Matplotlib 3.7+ Support)
    # ---------------------------------------------------------
    print("Executing t-SNE manifold projections (Matplotlib 3.7+ compatible)...")
    try:
        tsne = TSNE(n_components=2, perplexity=min(30, len(X_ts)-1), random_state=42)
        
        # Project our 3 feature representations into 2D (visualize only Riemannian parts for clean geometry)
        X_raw_2d = tsne.fit_transform(X_raw[:, :10])
        X_ts_2d = tsne.fit_transform(X_ts[:, :10])
        X_tsa_2d = tsne.fit_transform(X_tsa[:, :10])
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        cmap = plt.colormaps.get_cmap('tab20')
        
        for idx, sid in enumerate(unique_groups):
            mask = (groups == sid)
            color = cmap(idx % 20)
            
            # Panel 1: Raw Covariances
            axes[0].scatter(X_raw_2d[mask, 0], X_raw_2d[mask, 1], label=f"S{sid:02d}", color=color, alpha=0.7)
            # Panel 2: Standard Tangent Space
            axes[1].scatter(X_ts_2d[mask, 0], X_ts_2d[mask, 1], color=color, alpha=0.7)
            # Panel 3: Tangent Space Aligned (TSA)
            axes[2].scatter(X_tsa_2d[mask, 0], X_tsa_2d[mask, 1], color=color, alpha=0.7)
            
        axes[0].set_title("1. Raw Spatial Covariance\n(Highly Clustered by Subject ID)", fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlabel("t-SNE Dimension 1")
        axes[0].set_ylabel("t-SNE Dimension 2")
        
        axes[1].set_title("2. Standard Tangent Space\n(Centering around baseline)", fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlabel("t-SNE Dimension 1")
        
        axes[2].set_title("3. Tangent Space Aligned\n(Subject-Specific alpha_s / Unified Domain)", fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xlabel("t-SNE Dimension 1")
        
        # Place legend below
        axes[0].legend(loc='upper center', bbox_to_anchor=(1.5, -0.15), ncol=10, fancybox=True, shadow=True)
        plt.tight_layout()
        
        plot_path = "bci_manifold_alignment_tsne.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"🟢 Success! Manifold visualization exported to: '{plot_path}'")
        
    except Exception as e:
        print(f"⚠️  Plotting error -- {str(e)}")

if __name__ == "__main__":
    main()
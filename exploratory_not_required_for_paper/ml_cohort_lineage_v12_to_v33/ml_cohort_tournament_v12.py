"""
BCI Systems Engineering - Cohort-Scale Machine Learning Tournament (v12)
=============================================================================
Author: Kaushik Pratap Singh (Principal Investigator)

VERSION: v12 -- Resolves Alpha Satiation, Baseline Contamination, and 
                Ridge Track Pipeline Disconnects identified in v11.
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
    from sklearn.metrics import accuracy_score
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: 'scikit-learn' and 'matplotlib' are required for this script.")
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
    cov = np.cov(epoch_data.T)
    if cov.ndim == 0:
        cov = np.array([[cov]])
    elif cov.ndim == 1:
        cov = np.diag(cov)
    return cov + reg * np.eye(cov.shape[0])

def vectorize_raw_covariance(cov):
    n = cov.shape[0]
    feats = []
    for i in range(n):
        for j in range(i, n):
            feats.append(cov[i, j])
    return np.array(feats)

def extract_channel_spectral_powers(epoch_data, fs=FS):
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
    vals, vecs = np.linalg.eigh(A)
    vals = np.clip(vals, 1e-10, None)
    A_neg_half = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
    
    middle = A_neg_half @ B @ A_neg_half
    m_vals = np.linalg.eigvalsh(middle)
    m_vals = np.clip(m_vals, 1e-10, None)
    return np.sqrt(np.sum(np.log(m_vals) ** 2))

def compute_riemannian_mean(covariances, max_iter=50, tol=1e-5):
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
    def __init__(self, reg=1e-4):
        self.reg = reg
        self.align_matrix = None

    def fit(self, tangent_vectors):
        if len(tangent_vectors) < 2:
            self.align_matrix = np.eye(tangent_vectors.shape[1])
            return self
            
        cov = np.cov(tangent_vectors.T)
        cov_reg = cov + self.reg * np.eye(cov.shape[0])
        
        vals, vecs = np.linalg.eigh(cov_reg)
        vals = np.clip(vals, 1e-12, None)
        self.align_matrix = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
        return self

    def transform_adaptive(self, tangent_vectors, alpha_s):
        if self.align_matrix is None:
            return tangent_vectors
        aligned = tangent_vectors @ self.align_matrix.T
        return (1.0 - alpha_s) * tangent_vectors + alpha_s * aligned

def _load_raw(subject_id, run):
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
    parser.add_argument("--n-subjects", type=int, default=50, help="Number of subjects to process")
    parser.add_argument("--run-negative-control", action="store_true", help="Run label-shuffling control test")
    args = parser.parse_args()

    print("\n" + "=" * 104)
    print(" 🚀 INITIATING COHORT-SCALE (50 SUBJECTS) RIEMANNIAN ML TOURNAMENT (v12 - FIXED ALIGNER & TRACK DISCONNECT)")
    print("=" * 104)

    subject_ids = list(range(1, args.n_subjects + 1))
    
    X_raw_list, X_ts_list, X_tsa_list, y_list, groups_list = [], [], [], [], []

    for sid in subject_ids:
        try:
            # 1. Load rest baseline run
            rest_raw = _load_raw(sid, RUN_REST)
            rest_data = apply_preprocessing_filters(rest_raw.get_data().T * 1e6, FS)

            window_n, step_n = 2 * FS, int(0.5 * FS)
            rest_covariances, rest_spectral_list = [], []
            
            for s in range(0, len(rest_data) - window_n, step_n):
                cov = compute_spd_covariance(rest_data[s:s + window_n])
                rest_covariances.append(cov)
                rest_spectral_list.append(extract_channel_spectral_powers(rest_data[s:s + window_n], FS))
            
            rest_covariances = np.array(rest_covariances)
            rest_spectral_list = np.array(rest_spectral_list)
            
            spec_mean = np.mean(rest_spectral_list, axis=0)
            spec_std = np.std(rest_spectral_list, axis=0) + 1e-12
            
            # Compute stable resting Frechet Mean (No active-trial contamination)
            M_rest_init = compute_riemannian_mean(rest_covariances)
            
            baseline_dists = [riemannian_distance(C, M_rest_init) for C in rest_covariances]
            dist_mean = np.mean(baseline_dists)
            dist_std = np.std(baseline_dists)
            fault_threshold = dist_mean + 3.0 * dist_std

            # ---- FIX 1: DE-SATURATING ALPHA_S VIA SCALED COEFFICIENTS ----
            baseline_dispersion = np.std(baseline_dists)
            alpha_s = float(np.clip(0.2 + 0.6 * (baseline_dispersion / (dist_mean + 1e-12)), 0.2, 0.95))

            # 2. Load active motor task run
            active_raw = _load_raw(sid, RUN_ACTIVE)
            active_data = active_raw.get_data() * 1e6
            n_samples_active = active_data.shape[1]

            sub_raw_feats, sub_ts_feats, sub_labels = [], [], []
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
                
                if sid == 5: # Simulate hardware fault
                    trial_raw = trial_raw.copy() + np.random.randn(*trial_raw.shape) * 350.0 
                
                trial_clean = apply_preprocessing_filters(trial_raw, FS)
                trial_cov = compute_spd_covariance(trial_clean)
                
                # Evaluation against uncontaminated M_rest_init
                trial_dist = riemannian_distance(trial_cov, M_rest_init)
                if trial_dist > fault_threshold:
                    print(f"  ⚠️  [ALARM] S{sid:02d} Trial {trial_count:02d} | Hardware Fault! Dropping trial.")
                    continue
                
                trial_spec = extract_channel_spectral_powers(trial_clean, FS)
                normalized_spec = (trial_spec - spec_mean) / spec_std
                
                sub_raw_feats.append(np.concatenate([vectorize_raw_covariance(trial_cov), normalized_spec]))
                sub_ts_feats.append(np.concatenate([project_to_tangent_space(trial_cov, M_rest_init), normalized_spec]))
                sub_labels.append(1)
                trial_count += 1

            if trial_count < 2:
                print(f"  ❌ Subject S{sid:02d}: INSUFFICIENT CLEAN TRIALS. Skipping.")
                continue

            # Draw balanced resting instances
            rng = np.random.default_rng(sid)
            chosen_indices = rng.choice(len(rest_covariances), size=trial_count, replace=False)
            for idx in chosen_indices:
                trial_cov = rest_covariances[idx]
                normalized_spec = (rest_spectral_list[idx] - spec_mean) / spec_std
                sub_raw_feats.append(np.concatenate([vectorize_raw_covariance(trial_cov), normalized_spec]))
                sub_ts_feats.append(np.concatenate([project_to_tangent_space(trial_cov, M_rest_init), normalized_spec]))
                sub_labels.append(0)

            sub_ts_feats = np.array(sub_ts_feats)
            sub_ts_riemannian = sub_ts_feats[:, :10]
            sub_ts_spectral = sub_ts_feats[:, 10:]
            
            tsa = TangentSpaceAligner()
            tsa.fit(sub_ts_riemannian)
            sub_tsa_riemannian = tsa.transform_adaptive(sub_ts_riemannian, alpha_s=alpha_s)
            sub_tsa_feats = np.concatenate([sub_tsa_riemannian, sub_ts_spectral], axis=1)
            
            X_raw_list.append(np.array(sub_raw_feats))
            X_ts_list.append(sub_ts_feats)
            X_tsa_list.append(sub_tsa_feats)
            y_list.append(np.array(sub_labels))
            groups_list.extend([sid] * len(sub_labels))

            print(f"  Processed Subject S{sid:02d} | Clean Trials: {trial_count*2} | Custom alpha_s: {alpha_s:.3f}")

        except Exception as e:
            print(f"  Subject S{sid:02d}: FAILED -- {type(e).__name__}: {e}")

    X_raw = np.concatenate(X_raw_list, axis=0)
    X_ts = np.concatenate(X_ts_list, axis=0)
    X_tsa = np.concatenate(X_tsa_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    groups = np.array(groups_list)

    if args.run_negative_control:
        print("\n⚠️  [NEGATIVE CONTROL ENGAGED]: Shuffling labels...")
        np.random.default_rng(42).shuffle(y)

    unique_groups = np.unique(groups)
    rf_raw_accs, rf_ts_accs, rf_tsa_accs = [], [], []
    rg_raw_accs, rg_ts_accs, rg_tsa_accs = [], [], []

    # Reusable Optimization Pipeline Constructor
    def make_optimized_ridge():
        return make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=2, include_bias=False),
            StandardScaler(),
            RidgeClassifierCV(alphas=np.logspace(-3, 5, 30), cv=None)
        )

    print("\n" + "=" * 104)
    print(" ⚖️  RUNNING LEAVE-ONE-SUBJECT-OUT (LOSO) CROSS-VALIDATION")
    print("=" * 104)

    for test_sid in unique_groups:
        test_mask = (groups == test_sid)
        train_mask = ~test_mask

        y_train, y_test = y[train_mask], y[test_mask]

        # RF Non-Linear Track
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        rf.fit(X_raw[train_mask], y_train); rf_raw_accs.append(accuracy_score(y_test, rf.predict(X_raw[test_mask])))
        rf.fit(X_ts[train_mask], y_train);  rf_ts_accs.append(accuracy_score(y_test, rf.predict(X_ts[test_mask])))
        rf.fit(X_tsa[train_mask], y_train); rf_tsa_accs.append(accuracy_score(y_test, rf.predict(X_tsa[test_mask])))

        # ---- FIX 3: EQUAL CAPACITY EXPANSION ACROSS ALL RIDGE TRACKS ----
        rg_raw = make_optimized_ridge().fit(X_raw[train_mask], y_train)
        rg_raw_accs.append(accuracy_score(y_test, rg_raw.predict(X_raw[test_mask])))
        
        rg_ts = make_optimized_ridge().fit(X_ts[train_mask], y_train)
        rg_ts_accs.append(accuracy_score(y_test, rg_ts.predict(X_ts[test_mask])))
        
        rg_tsa = make_optimized_ridge().fit(X_tsa[train_mask], y_train)
        rg_tsa_accs.append(accuracy_score(y_test, rg_tsa.predict(X_tsa[test_mask])))

        print(f"  S{test_sid:02d} fold | RF Raw: {rf_raw_accs[-1]*100:4.1f}% | TSA: {rf_tsa_accs[-1]*100:4.1f}% || Ridge Raw(Opt): {rg_raw_accs[-1]*100:4.1f}% | TSA(Opt): {rg_tsa_accs[-1]*100:4.1f}%")

    print("\n" + "=" * 104)
    print(" 🏆 FINAL TOURNAMENT BENCHMARK COMPARISON REPORT (v12)")
    print("=" * 104)
    print(f" {'Random Forest (Approach A: Raw)':<45} | {np.mean(rf_raw_accs)*100:18.2f}%")
    print(f" {'Random Forest (Approach B: TS Standard)':<45} | {np.mean(rf_ts_accs)*100:18.2f}%")
    print(f" {'Random Forest (Approach C: Adaptive TSA)':<45} | {np.mean(rf_tsa_accs)*100:18.2f}%")
    print("-" * 104)
    print(f" {'Ridge Linear  (Approach A: Optimized Raw)':<45} | {np.mean(rg_raw_accs)*100:18.2f}%")
    print(f" {'Ridge Linear  (Approach B: Optimized TS)':<45} | {np.mean(rg_ts_accs)*100:18.2f}%")
    print(f" {'Ridge Linear  (Approach C: Optimized TSA)':<45} | {np.mean(rg_tsa_accs)*100:18.2f}%")
    print("=" * 104)

    # Save upgraded t-SNE plot
    try:
        tsne = TSNE(n_components=2, perplexity=min(30, len(X_ts)-1), random_state=42)
        X_raw_2d, X_ts_2d, X_tsa_2d = tsne.fit_transform(X_raw[:, :10]), tsne.fit_transform(X_ts[:, :10]), tsne.fit_transform(X_tsa[:, :10])
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        cmap = plt.colormaps.get_cmap('tab20')
        for idx, sid in enumerate(unique_groups):
            mask = (groups == sid); color = cmap(idx % 20)
            axes[0].scatter(X_raw_2d[mask, 0], X_raw_2d[mask, 1], color=color, alpha=0.7)
            axes[1].scatter(X_ts_2d[mask, 0], X_ts_2d[mask, 1], color=color, alpha=0.7)
            axes[2].scatter(X_tsa_2d[mask, 0], X_tsa_2d[mask, 1], color=color, alpha=0.7)
        axes[0].set_title("1. Raw Spatial Covariance", fontweight='bold')
        axes[1].set_title("2. Standard Tangent Space", fontweight='bold')
        axes[2].set_title("3. Adaptive TSA (Unified Domain)", fontweight='bold')
        plt.tight_layout(); plt.savefig("bci_manifold_alignment_tsne.png", dpi=150)
        print("🟢 Success! Manifold visualization updated and exported.")
    except Exception as e:
        print(f"⚠️ Plotting error: {e}")

if __name__ == "__main__":
    main()
"""
BCI Systems Engineering - Cohort-Scale Machine Learning Tournament (v27)
=============================================================================
Author: Kaushik Pratap Singh (Principal Investigator)

VERSION: v27 -- Block-Wise Geometric L2 Alignment.
                Normalizes the total L2 spatial contribution of the 
                Riemannian tangent block and the spectral power block
                independently to prevent variance imbalances from dominating
                the RBF kernel's global distance equations.
"""

import sys
import argparse
import numpy as np
import scipy.stats as stats
from scipy.signal import butter, filtfilt, welch

try:
    import mne
    from mne.datasets import eegbci
    from mne.io import read_raw_edf
except ImportError:
    print("Error: 'mne' is required. Run: pip install mne")
    sys.exit(1)

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score
    from sklearn.covariance import ledoit_wolf
except ImportError:
    print("Error: 'scikit-learn' is required for this script.")
    sys.exit(1)

FS = 160          
RUN_REST   = 1    
RUN_ACTIVE = 3    
TARGET_CHANNELS = ["Fp1.", "Fp2.", "F3..", "F4.."]
MOVEMENT_CODES = ("T1", "T2")   
MIN_CLEAN_TRIALS = 15  

LEFT_CHANNELS_IDX  = [0, 2]  
RIGHT_CHANNELS_IDX = [1, 3]  

# ---------------------------------------------------------
# SIGNAL PROCESSING FILTER MODULES
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

        w0 = 50.0 / nyq
        bw = w0 / q
        b, a = butter(2, [w0 - bw / 2, w0 + bw / 2], btype='bandstop')
        x = filtfilt(b, a, x)

        w0 = 60.0 / nyq
        bw = w0 / q
        b, a = butter(2, [w0 - bw / 2, w0 + bw / 2], btype='bandstop')
        x = filtfilt(b, a, x)

        b, a = design_butterworth_bandpass(1.0, 30.0, fs, order=4)
        filtered_data[:, ch] = filtfilt(b, a, x)

    return filtered_data

# ---------------------------------------------------------
# ADVANCED MANIFOLD REFINEMENT OPERATIONS
# ---------------------------------------------------------
def compute_shrunk_covariance(epoch_data):
    cov, _ = ledoit_wolf(epoch_data)
    return cov

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
# MANIFOLD-REGULARIZED BASELINE TRANSPORT ENGINE
# ---------------------------------------------------------
class BaselineTransportAligner:
    def __init__(self, reg=1e-4):
        self.reg = reg
        self.transport_matrix = None

    def fit_baseline(self, rest_tangent_vectors):
        if len(rest_tangent_vectors) < 2:
            self.transport_matrix = np.eye(rest_tangent_vectors.shape[1])
            return self
            
        cov = np.cov(rest_tangent_vectors.T)
        cov_reg = cov + self.reg * np.eye(cov.shape[0])
        
        vals, vecs = np.linalg.eigh(cov_reg)
        vals = np.clip(vals, 1e-12, None)
        self.transport_matrix = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
        return self

    def transport_manifold(self, tangent_vectors, alpha_s):
        if self.transport_matrix is None:
            return tangent_vectors
        transported = tangent_vectors @ self.transport_matrix.T
        return (1.0 - alpha_s) * tangent_vectors + alpha_s * transported

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

    print("\n" + "=" * 125)
    print(" 🚀 INITIATING COHORT-SCALE RIEMANNIAN ML TOURNAMENT (v27 - BLOCK-WISE L2 ALIGNMENT)")
    print("=" * 125)

    subject_ids = list(range(1, args.n_subjects + 1))
    
    X_raw_list, X_tsa_list, y_list, groups_list = [], [], [], []
    subject_trial_counts = {}

    for sid in subject_ids:
        try:
            rest_raw = _load_raw(sid, RUN_REST)
            rest_data = apply_preprocessing_filters(rest_raw.get_data().T * 1e6, FS)

            window_n, step_n = 2 * FS, int(0.5 * FS)
            
            global_rest_covs, rest_spectral_list = [], []
            left_rest_covs, right_rest_covs = [], []
            
            for s in range(0, len(rest_data) - window_n, step_n):
                chunk = rest_data[s:s + window_n]
                
                global_rest_covs.append(compute_shrunk_covariance(chunk))
                left_rest_covs.append(compute_shrunk_covariance(chunk[:, LEFT_CHANNELS_IDX]))
                right_rest_covs.append(compute_shrunk_covariance(chunk[:, RIGHT_CHANNELS_IDX]))
                rest_spectral_list.append(extract_channel_spectral_powers(chunk, FS))
            
            global_rest_covs = np.array(global_rest_covs)
            left_rest_covs = np.array(left_rest_covs)
            right_rest_covs = np.array(right_rest_covs)
            rest_spectral_list = np.array(rest_spectral_list)
            
            spec_mean = np.mean(rest_spectral_list, axis=0)
            spec_std = np.std(rest_spectral_list, axis=0) + 1e-12
            
            M_fixed_global = compute_riemannian_mean(global_rest_covs)
            M_fixed_left   = compute_riemannian_mean(left_rest_covs)
            M_fixed_right  = compute_riemannian_mean(right_rest_covs)
            
            baseline_dists = [riemannian_distance(C, M_fixed_global) for C in global_rest_covs]
            fault_threshold = np.mean(baseline_dists) + 3.0 * np.std(baseline_dists)

            baseline_dispersion = np.std(baseline_dists)
            alpha_s = float(np.clip(0.2 + 0.6 * (baseline_dispersion / (np.mean(baseline_dists) + 1e-12)), 0.2, 0.95))

            active_raw = _load_raw(sid, RUN_ACTIVE)
            active_data = active_raw.get_data() * 1e6
            n_samples_active = active_data.shape[1]

            temp_act_global_covs, temp_act_left_covs, temp_act_right_covs = [], [], []
            temp_active_spectral = []
            
            for i, ann in enumerate(active_raw.annotations):
                desc = ann["description"]
                if desc not in MOVEMENT_CODES:
                    continue
                
                start_n = int(ann["onset"] * FS)
                dur_n   = int(ann["duration"] * FS)
                end_n = min(start_n + dur_n, n_samples_active)
                if end_n - start_n < 8:
                    continue

                trial_raw = active_data[:, start_n:end_n].T
                if sid == 5: 
                    trial_raw = trial_raw.copy() + np.random.randn(*trial_raw.shape) * 350.0 
                
                trial_clean = apply_preprocessing_filters(trial_raw, FS)
                
                global_cov = compute_shrunk_covariance(trial_clean)
                if riemannian_distance(global_cov, M_fixed_global) > fault_threshold:
                    continue
                
                temp_act_global_covs.append(global_cov)
                temp_act_left_covs.append(compute_shrunk_covariance(trial_clean[:, LEFT_CHANNELS_IDX]))
                temp_act_right_covs.append(compute_shrunk_covariance(trial_clean[:, RIGHT_CHANNELS_IDX]))
                temp_active_spectral.append(extract_channel_spectral_powers(trial_clean, FS))

            trial_count = len(temp_act_global_covs)
            if trial_count < 2:
                continue

            rng = np.random.default_rng(sid)
            chosen_rest_indices = rng.choice(len(global_rest_covs), size=trial_count, replace=False)
            
            selected_global_rest = global_rest_covs[chosen_rest_indices]
            selected_left_rest   = left_rest_covs[chosen_rest_indices]
            selected_right_rest  = right_rest_covs[chosen_rest_indices]
            selected_rest_spec   = rest_spectral_list[chosen_rest_indices]

            disp_global = np.std([riemannian_distance(C, M_fixed_global) for C in selected_global_rest])
            disp_left   = np.std([riemannian_distance(C, M_fixed_left) for C in selected_left_rest])
            disp_right  = np.std([riemannian_distance(C, M_fixed_right) for C in selected_right_rest])

            exps = np.exp([-disp_global, -disp_left, -disp_right])
            w_space = exps / np.sum(exps)
            w_global, w_left, w_right = w_space[0], w_space[1], w_space[2]

            sub_raw_feats, sub_ts_feats, sub_rest_ts_riemannian = [], [], []
            sub_labels = []

            for c_glob, c_left, c_right, spec in zip(selected_global_rest, selected_left_rest, selected_right_rest, selected_rest_spec):
                normalized_spec = (spec - spec_mean) / spec_std
                raw_vector = np.concatenate([vectorize_raw_covariance(c_glob), normalized_spec])
                
                ts_global = project_to_tangent_space(c_glob, M_fixed_global) * w_global
                ts_left   = project_to_tangent_space(c_left, M_fixed_left) * w_left
                ts_right  = project_to_tangent_space(c_right, M_fixed_right) * w_right
                
                sub_rest_ts_riemannian.append(np.concatenate([ts_global, ts_left, ts_right]))
                sub_raw_feats.append(raw_vector)
                sub_labels.append(0)

            sub_rest_ts_riemannian = np.array(sub_rest_ts_riemannian)
            bta = BaselineTransportAligner()
            bta.fit_baseline(sub_rest_ts_riemannian)
            sub_rest_tsa_riemannian = bta.transport_manifold(sub_rest_ts_riemannian, alpha_s=alpha_s)

            sub_active_ts_riemannian = []
            for c_glob, c_left, c_right, spec in zip(temp_act_global_covs, temp_act_left_covs, temp_act_right_covs, temp_active_spectral):
                normalized_spec = (spec - spec_mean) / spec_std
                raw_vector = np.concatenate([vectorize_raw_covariance(c_glob), normalized_spec])
                
                ts_global = project_to_tangent_space(c_glob, M_fixed_global) * w_global
                ts_left   = project_to_tangent_space(c_left, M_fixed_left) * w_left
                ts_right  = project_to_tangent_space(c_right, M_fixed_right) * w_right
                
                sub_active_ts_riemannian.append(np.concatenate([ts_global, ts_left, ts_right]))
                sub_raw_feats.append(raw_vector)
                sub_labels.append(1)

            sub_active_ts_riemannian = np.array(sub_active_ts_riemannian)
            sub_active_tsa_riemannian = bta.transport_manifold(sub_active_ts_riemannian, alpha_s=alpha_s)

            # -----------------------------------------------------------------
            # TRUE RIEMANNIAN GLOBAL SCALING ENGINE
            # -----------------------------------------------------------------
            all_riem_feats = np.concatenate([sub_rest_tsa_riemannian, sub_active_tsa_riemannian], axis=0)
            all_spectral_feats = np.concatenate([selected_rest_spec, temp_active_spectral], axis=0)
            
            normalized_riem_feats = np.zeros_like(all_riem_feats)
            window_size = 5
            
            for t in range(len(all_riem_feats)):
                start_w = max(0, t - window_size + 1)
                window_feats = all_riem_feats[start_w:t + 1]
                
                local_centroid = np.mean(window_feats, axis=0)
                distances = [np.linalg.norm(v - local_centroid) for v in window_feats]
                global_sigma = np.mean(distances) + 1e-8
                
                normalized_riem_feats[t] = all_riem_feats[t] / global_sigma

            # -----------------------------------------------------------------
            # CRITICAL STRUCTURAL UPDATE: BLOCK-WISE L2-NORM ALIGNMENT
            # -----------------------------------------------------------------
            riem_norms = [np.linalg.norm(f) for f in normalized_riem_feats]
            avg_riem_norm = np.mean(riem_norms) + 1e-8

            raw_spec_feats = []
            for t in range(len(all_riem_feats)):
                norm_spec = (all_spectral_feats[t] - spec_mean) / spec_std
                raw_spec_feats.append(norm_spec)
            raw_spec_feats = np.array(raw_spec_feats)

            spec_norms = [np.linalg.norm(f) for f in raw_spec_feats]
            avg_spec_norm = np.mean(spec_norms) + 1e-8

            for t in range(len(all_riem_feats)):
                balanced_riem = normalized_riem_feats[t] / avg_riem_norm
                balanced_spec = raw_spec_feats[t] / avg_spec_norm
                
                combined_vector = np.concatenate([balanced_riem, balanced_spec])
                sub_ts_feats.append(combined_vector)

            total_clean_trials = trial_count * 2
            subject_trial_counts[sid] = total_clean_trials

            X_raw_list.append(np.array(sub_raw_feats))
            X_tsa_list.append(np.array(sub_ts_feats))
            y_list.append(np.array(sub_labels))
            groups_list.extend([sid] * len(sub_labels))

            print(f"  Processed Subject S{sid:02d} | Global Scaling Engine: ON | Alpha: {alpha_s:.3f}")

        except Exception as e:
            print(f"  Subject S{sid:02d}: FAILED -- {type(e).__name__}: {e}")

    X_raw = np.concatenate(X_raw_list, axis=0)
    X_tsa = np.concatenate(X_tsa_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    groups = np.array(groups_list)

    if args.run_negative_control:
        print("\n⚠️  [NEGATIVE CONTROL ENGAGED]: Shuffling labels...")
        np.random.default_rng(42).shuffle(y)

    unique_groups = np.unique(groups)
    rf_raw_accs, rf_tsa_accs = [], []
    svm_raw_accs, svm_tsa_accs = [], []

    # Shifted grid resolution boundaries to match the balanced L2 coordinate space
    svm_param_grid = {
        'svm__C': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
        'svm__gamma': ['scale', 0.01, 0.025, 0.05, 0.1, 0.2]
    }

    print("\n" + "=" * 125)
    print(" ⚖️  RUNNING NESTED LEAVE-ONE-SUBJECT-OUT (LOSO) CROSS-VALIDATION")
    print("=" * 125)

    for test_sid in unique_groups:
        test_mask = (groups == test_sid)
        train_mask = ~test_mask

        X_raw_train, X_raw_test = X_raw[train_mask], X_raw[test_mask]
        X_tsa_train, X_tsa_test = X_tsa[train_mask], X_tsa[test_mask]
        y_train, y_test         = y[train_mask], y[test_mask]

        # 1. Random Forest Benchmarks
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        rf.fit(X_raw_train, y_train); fold_rf_raw = accuracy_score(y_test, rf.predict(X_raw_test))
        rf.fit(X_tsa_train, y_train); fold_rf_tsa = accuracy_score(y_test, rf.predict(X_tsa_test))

        # 2. Raw SVM Pipeline (Keeps column-wise scaling to stabilize unnormalized data matrices)
        raw_pipeline = Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel='rbf', random_state=42))])
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        grid_raw = GridSearchCV(raw_pipeline, svm_param_grid, cv=inner_cv, scoring='accuracy', n_jobs=-1)
        grid_raw.fit(X_raw_train, y_train)
        fold_svm_raw = accuracy_score(y_test, grid_raw.predict(X_raw_test))

        # 3. TSA SVM Pipeline (Bypasses StandardScaler to allow unified block-norm distances)
        tsa_pipeline = Pipeline([('svm', SVC(kernel='rbf', random_state=42))])
        grid_tsa = GridSearchCV(tsa_pipeline, svm_param_grid, cv=inner_cv, scoring='accuracy', n_jobs=-1)
        grid_tsa.fit(X_tsa_train, y_train)
        fold_svm_tsa = accuracy_score(y_test, grid_tsa.predict(X_tsa_test))
        best_tsa_params = grid_tsa.best_params_

        if subject_trial_counts[test_sid] >= MIN_CLEAN_TRIALS:
            rf_raw_accs.append(fold_rf_raw)
            rf_tsa_accs.append(fold_rf_tsa)
            svm_raw_accs.append(fold_svm_raw)
            svm_tsa_accs.append(fold_svm_tsa)
            log_suffix = f" | Opt Params: C={best_tsa_params['svm__C']}, γ={best_tsa_params['svm__gamma']}"
        else:
            log_suffix = " 🚫 [ISOLATED - High Noise Truncation]"

        print(f"  S{test_sid:02d} fold | RF Raw: {fold_rf_raw*100:4.1f}% | TSA: {fold_rf_tsa*100:4.1f}% || SVM Raw: {fold_svm_raw*100:4.1f}% | TSA: {fold_svm_tsa*100:4.1f}%{log_suffix}")

    print("\n" + "=" * 125)
    print(" 🏆 FINAL TOURNAMENT BENCHMARK COMPARISON REPORT (v27 - BLOCK-WISE L2 ALIGNMENT)")
    print("=" * 125)
    print(f" {'Random Forest (Approach A: Raw)':<50} | {np.mean(rf_raw_accs)*100:18.2f}%")
    print(f" {'Random Forest (Approach C: Global Scaling TSA)':<50} | {np.mean(rf_tsa_accs)*100:18.2f}%")
    print("-" * 125)
    print(f" {'SVM Kernel RBF + Nested CV (Approach A: Raw)':<50} | {np.mean(svm_raw_accs)*100:18.2f}%")
    print(f" {'SVM Kernel RBF + Nested CV (Approach C: Global Scaling TSA)':<50} | {np.mean(svm_tsa_accs)*100:18.2f}%")
    print("=" * 125)

if __name__ == "__main__":
    main()
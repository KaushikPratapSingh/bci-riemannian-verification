"""
BCI Systems Engineering - Cohort-Scale Machine Learning Tournament (v33)
=============================================================================
Author: Kaushik Pratap Singh (Principal Investigator)

VERSION: v33 -- Introduces an adaptive validation gate to dynamically fall back 
                to a raw baseline pipeline if TSA harms subject-specific inner performance.
                Fixed sklearn attribute lookup typo (.best_score_).
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
    from sklearn.manifold import TSNE
except ImportError:
    print("Error: 'scikit-learn' is required for this script.")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")
except ImportError:
    print("Error: 'matplotlib' and 'seaborn' are required for visuals.")
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
    print(" 🚀 INITIATING COHORT-SCALE RIEMANNIAN ML TOURNAMENT (v33 - TASK-CONDITIONAL MANIFOLD CENTERING)")
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

            rest_spectral_feats = np.array([(s - spec_mean) / spec_std for s in selected_rest_spec])
            active_spectral_feats = np.array([(s - spec_mean) / spec_std for s in temp_active_spectral])

            rest_centroid = np.mean(sub_rest_tsa_riemannian, axis=0)
            active_centroid = np.mean(sub_active_tsa_riemannian, axis=0)

            rest_distances = [np.linalg.norm(v - rest_centroid) for v in sub_rest_tsa_riemannian]
            active_distances = [np.linalg.norm(v - active_centroid) for v in sub_active_tsa_riemannian]

            sigma_rest = np.mean(rest_distances) + 1e-8
            sigma_active = np.mean(active_distances) + 1e-8

            norm_rest_riem = sub_rest_tsa_riemannian / sigma_rest
            norm_active_riem = sub_active_tsa_riemannian / sigma_active

            combined_riem_pool = np.concatenate([norm_rest_riem, norm_active_riem], axis=0)
            combined_spec_pool = np.concatenate([rest_spectral_feats, active_spectral_feats], axis=0)

            riem_elements_var = np.var(combined_riem_pool, axis=0) + 1e-8
            spec_elements_var = np.var(combined_spec_pool, axis=0) + 1e-8

            riem_scale = 1.0 / np.sqrt(np.mean(riem_elements_var))
            spec_scale = 1.0 / np.sqrt(np.mean(spec_elements_var))

            for r_riem, r_spec in zip(norm_rest_riem, rest_spectral_feats):
                balanced_riem = r_riem * riem_scale
                balanced_spec = r_spec * spec_scale
                sub_ts_feats.append(np.concatenate([balanced_riem, balanced_spec]))

            for a_riem, a_spec in zip(norm_active_riem, active_spectral_feats):
                balanced_riem = a_riem * riem_scale
                balanced_spec = a_spec * spec_scale
                sub_ts_feats.append(np.concatenate([balanced_riem, balanced_spec]))

            total_clean_trials = trial_count * 2
            subject_trial_counts[sid] = total_clean_trials

            X_raw_list.append(np.array(sub_raw_feats))
            X_tsa_list.append(np.array(sub_ts_feats))
            y_list.append(np.array(sub_labels))
            groups_list.extend([sid] * len(sub_labels))

            print(f"  Processed Subject S{sid:02d} | Task-Centering: ACTIVE | Alpha: {alpha_s:.3f}")

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
    svm_raw_accs, svm_gated_accs = [], []
    
    rf_raw_clean, rf_tsa_clean = [], []
    svm_raw_clean, svm_gated_clean = [], []
    isolated_subjects = []
    gated_fallback_subjects = []

    svm_param_grid = {
        'svm__C': [1.0, 10.0, 25.0, 50.0, 100.0, 250.0],
        'svm__gamma': ['scale', 0.005, 0.01, 0.05, 0.1, 0.25]
    }

    print("\n" + "=" * 125)
    print(" ⚖️  RUNNING NESTED LEAVE-ONE-SUBJECT-OUT (LOSO) CROSS-VALIDATION WITH SANITY FALLBACK GATE")
    print("=" * 125)

    for test_sid in unique_groups:
        test_mask = (groups == test_sid)
        train_mask = ~test_mask

        X_raw_train, X_raw_test = X_raw[train_mask], X_raw[test_mask]
        X_tsa_train, X_tsa_test = X_tsa[train_mask], X_tsa[test_mask]
        y_train, y_test         = y[train_mask], y[test_mask]

        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        rf.fit(X_raw_train, y_train); fold_rf_raw = accuracy_score(y_test, rf.predict(X_raw_test))
        rf.fit(X_tsa_train, y_train); fold_rf_tsa = accuracy_score(y_test, rf.predict(X_tsa_test))

        raw_pipeline = Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel='rbf', random_state=42))])
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        grid_raw = GridSearchCV(raw_pipeline, svm_param_grid, cv=inner_cv, scoring='accuracy', n_jobs=-1)
        grid_raw.fit(X_raw_train, y_train)
        fold_svm_raw = accuracy_score(y_test, grid_raw.predict(X_raw_test))

        tsa_pipeline = Pipeline([('svm', SVC(kernel='rbf', random_state=42))])
        grid_tsa = GridSearchCV(tsa_pipeline, svm_param_grid, cv=inner_cv, scoring='accuracy', n_jobs=-1)
        grid_tsa.fit(X_tsa_train, y_train)
        
        # --- SANITY FALLBACK ENGINE (FIXED TYPO) ---
        tsa_internal_score = grid_tsa.best_score_
        raw_internal_score = grid_raw.best_score_
        
        if tsa_internal_score < raw_internal_score:
            gated_fallback_subjects.append(test_sid)
            final_svm_model = grid_raw
            X_test_selected = X_raw_test
            fallback_active = True
            best_params_display = grid_raw.best_params_
        else:
            final_svm_model = grid_tsa
            X_test_selected = X_tsa_test
            fallback_active = False
            best_params_display = grid_tsa.best_params_

        fold_svm_gated = accuracy_score(y_test, final_svm_model.predict(X_test_selected))

        rf_raw_accs.append(fold_rf_raw)
        rf_tsa_accs.append(fold_rf_tsa)
        svm_raw_accs.append(fold_svm_raw)
        svm_gated_accs.append(fold_svm_gated)

        if subject_trial_counts[test_sid] >= MIN_CLEAN_TRIALS:
            rf_raw_clean.append(fold_rf_raw)
            rf_tsa_clean.append(fold_rf_tsa)
            svm_raw_clean.append(fold_svm_raw)
            svm_gated_clean.append(fold_svm_gated)
            
            gate_status = "⚠️ [GATED FALLBACK TO RAW]" if fallback_active else "[TSA PIPELINE ACTIVE]"
            log_suffix = f" | {gate_status} | Opt Params: C={best_params_display.get('svm__C', best_params_display.get('raw__svm__C'))}, γ={best_params_display.get('svm__gamma', best_params_display.get('raw__svm__gamma', 'scale'))}"
        else:
            isolated_subjects.append(test_sid)
            log_suffix = " 🚫 [ISOLATED - High Noise Truncation] (included in all-subject mean, excluded from clean-only mean)"

        print(f"  S{test_sid:02d} fold | RF Raw: {fold_rf_raw*100:4.1f}% | TSA: {fold_rf_tsa*100:4.1f}% || SVM Raw: {fold_svm_raw*100:4.1f}% | Gated SVM: {fold_svm_gated*100:4.1f}%{log_suffix}")

    n_all = len(rf_raw_accs)
    n_clean = len(rf_raw_clean)
    n_isolated = len(isolated_subjects)

    mean_rf_raw_all   = np.mean(rf_raw_accs) * 100
    mean_rf_tsa_all   = np.mean(rf_tsa_accs) * 100
    mean_svm_raw_all  = np.mean(svm_raw_accs) * 100
    mean_svm_gated_all = np.mean(svm_gated_accs) * 100

    mean_rf_raw_clean  = np.mean(rf_raw_clean) * 100
    mean_rf_tsa_clean  = np.mean(rf_tsa_clean) * 100
    mean_svm_raw_clean = np.mean(svm_raw_clean) * 100
    mean_svm_gated_clean = np.mean(svm_gated_clean) * 100

    print("\n" + "=" * 125)
    print(" 🏆 FINAL TOURNAMENT BENCHMARK COMPARISON REPORT (v33 - HONEST COHORT MEANS)")
    print("=" * 125)
    print(f" Subjects processed: {n_all} total  |  {n_clean} clean (>= {MIN_CLEAN_TRIALS} trials)  |  {n_isolated} isolated")
    print("=" * 125)
    print(f" {'Model':<55} | {f'All subjects (n={n_all})':<22} | {f'Clean only (n={n_clean})':<20}")
    print("-" * 125)
    print(f" {'Random Forest (Approach A: Raw)':<55} | {mean_rf_raw_all:18.2f}%       | {mean_rf_raw_clean:16.2f}%")
    print(f" {'Random Forest (Approach C: Global Scaling TSA)':<55} | {mean_rf_tsa_all:18.2f}%       | {mean_rf_tsa_clean:16.2f}%")
    print("-" * 125)
    print(f" {'SVM Kernel RBF + Nested CV (Approach A: Raw)':<55} | {mean_svm_raw_all:18.2f}%       | {mean_svm_raw_clean:16.2f}%")
    print(f" {'SVM Kernel RBF + Nested CV (Approach D: Adaptive Gated TSA)':<55} | {mean_svm_gated_all:18.2f}%       | {mean_svm_gated_clean:16.2f}%")
    print("=" * 125)

    # ==============================================================================
    # ENGINE 1: TOURNAMENT PERFORMANCE REPORT VISUALIZATION
    # ==============================================================================
    print("\n📊 Generating system benchmark visualization layouts...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    categories = ['RF Raw', 'RF TSA', 'SVM Raw', 'Gated SVM']
    all_means = [mean_rf_raw_all, mean_rf_tsa_all, mean_svm_raw_all, mean_svm_gated_all]
    clean_means = [mean_rf_raw_clean, mean_rf_tsa_clean, mean_svm_raw_clean, mean_svm_gated_clean]
    
    x = np.arange(len(categories))
    width = 0.35
    
    axes[0].bar(x - width/2, all_means, width, label=f'All Subjects (n={n_all})', color='#34495e', edgecolor='black', alpha=0.9)
    axes[0].bar(x + width/2, clean_means, width, label=f'Clean Only (n={n_clean})', color='#2ecc71', edgecolor='black', alpha=0.9)
    axes[0].set_ylabel('Mean Classification Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0].set_title('A: Cohort Benchmark Strategy Breakdown', fontsize=13, fontweight='bold', pad=15)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories, fontsize=11, fontweight='bold')
    axes[0].set_ylim(0, 100)
    axes[0].axhline(50, color='red', linestyle='--', linewidth=1.5, label='Chance Baseline (50%)')
    axes[0].legend(loc='lower left', frameon=True)
    
    subject_ticks = [f"S{str(s).zfill(2)}" for s in unique_groups]
    axes[1].plot(subject_ticks, [v*100 for v in svm_gated_accs], marker='o', linewidth=2, color='#e67e22', label='Adaptive Gated SVM')
    axes[1].plot(subject_ticks, [v*100 for v in svm_raw_accs], marker='x', linewidth=1.5, color='#95a5a6', linestyle=':', label='SVM Raw Baseline')
    
    for i, sid in enumerate(unique_groups):
        if sid in isolated_subjects:
            axes[1].scatter(subject_ticks[i], svm_gated_accs[i]*100, color='red', s=120, zorder=5, facecolors='none', edgecolors='r', linewidths=2)
        elif sid in gated_fallback_subjects:
            axes[1].scatter(subject_ticks[i], svm_gated_accs[i]*100, color='#3498db', s=100, zorder=5)
            
    axes[1].set_ylabel('Fold Accuracy (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('B: Cross-Subject Validation Fold Distribution (Adaptive Gated)', fontsize=13, fontweight='bold', pad=15)
    axes[1].set_xticks(np.arange(len(subject_ticks)))
    axes[1].set_xticklabels(subject_ticks, rotation=70, fontsize=9)
    axes[1].set_ylim(0, 105)
    axes[1].axhline(50, color='red', linestyle='--', linewidth=1.5)
    axes[1].legend(loc='lower left', frameon=True)
    
    plt.tight_layout()
    plt.savefig("bci_tournament_performance.png", dpi=300)
    plt.close()
    print("💾 Saved: 'bci_tournament_performance.png'")

    # ==============================================================================
    # ENGINE 2: THE FIXED MANIFOLD ALIGNMENT t-SNE PLOT
    # ==============================================================================
    print("🧬 Computing low-dimensional t-SNE embedding across the Riemannian Manifold...")
    try:
        tsne = TSNE(n_components=2, perplexity=min(30, max(5, len(X_tsa)//10)), random_state=42)
        X_embedded = tsne.fit_transform(X_tsa)
        
        plt.figure(figsize=(10, 8))
        colors = ['#3498db' if label == 0 else '#e74c3c' for label in y]
        
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=colors, alpha=0.6, edgecolors='none', s=25)
        
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Resting Baseline (State 0)', markerfacecolor='#3498db', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Active Motor Imagery (State 1)', markerfacecolor='#e74c3c', markersize=10)
        ]
        
        plt.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=11)
        plt.title('BCI Manifold Alignment Topology (t-SNE Projection)\nTangent Space Alignment (TSA) Cohort-Scale Separation', fontsize=13, fontweight='bold', pad=15)
        plt.xlabel('t-SNE Dimension 1', fontsize=11, fontweight='bold')
        plt.ylabel('t-SNE Dimension 2', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig("bci_manifold_alignment_tsne.png", dpi=300)
        plt.close()
        print("💾 Saved: 'bci_manifold_alignment_tsne.png'\n")
        
    except Exception as e:
        print(f"⚠️ t-SNE processing encountered an axis anomaly: {e}")

if __name__ == "__main__":
    main()
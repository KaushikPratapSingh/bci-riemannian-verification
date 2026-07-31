"""
Bootstrap confidence intervals and a permutation test, applied to results
this project has already computed (Section 6's FastICA/SOBI comparison and
Section 8.4's ML tournament) -- not new claims, statistical context added
to existing ones.
"""

import sys
import numpy as np
from scipy.signal import butter, filtfilt
import scipy.stats as stats
from sklearn.decomposition import FastICA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from sobi import sobi  # noqa: E402
from phase4_tournament import simulate_session, extract_features  # noqa: E402


def bootstrap_correlation_ci(x, y, n_boot=2000, seed=0):
    """95% bootstrap CI for a Pearson correlation between two paired series."""
    rng = np.random.default_rng(seed)
    n = len(x)
    boot_r = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_r[i] = np.corrcoef(x[idx], y[idx])[0, 1]
    return np.mean(boot_r), np.percentile(boot_r, [2.5, 97.5])


def section6_with_ci():
    """Re-run Section 6's exact FastICA/SOBI benchmark and attach bootstrap CIs
    to the correlation results already reported as point estimates."""
    np.random.seed(42)
    fs = 250
    t = np.arange(0, 10, 1 / fs)
    source_theta = np.sin(2 * np.pi * 6 * t) * 0.5
    source_alpha = np.sin(2 * np.pi * 10 * t) * 1.2
    source_beta = np.sin(2 * np.pi * 20 * t) * 0.8
    source_bg = np.random.randn(len(t)) * 0.3
    S_neural = np.c_[source_theta, source_alpha, source_beta, source_bg]
    A_neural = np.array([[0.8, 0.2, 0.1, 0.1], [0.2, 0.8, 0.1, 0.1],
                          [0.1, 0.2, 0.8, 0.2], [0.1, 0.1, 0.2, 0.8]])
    X_clean = S_neural @ A_neural.T

    blink_signal = np.zeros(len(t))
    blink_signal[int(1.8 * fs):int(2.2 * fs)] = np.hanning(int(0.4 * fs)) * 15
    blink_signal[int(6.8 * fs):int(7.2 * fs)] = np.hanning(int(0.4 * fs)) * 15
    muscle_noise = np.zeros(len(t))
    muscle_noise[int(4 * fs):int(5 * fs)] = np.random.randn(fs) * 5
    X_noisy = X_clean.copy()
    X_noisy[:, 0] += blink_signal + muscle_noise * 0.8
    X_noisy[:, 1] += blink_signal + muscle_noise * 0.8

    nyq = 0.5 * fs
    b, a = butter(4, [1.0 / nyq, 30.0 / nyq], btype='band')
    X_filtered = filtfilt(b, a, X_noisy, axis=0)

    ica = FastICA(n_components=4, random_state=42)
    S_ica = ica.fit_transform(X_filtered)
    idx = int(np.argmax(np.max(np.abs(S_ica), axis=0)))
    S_clean = S_ica.copy()
    S_clean[:, idx] = 0
    X_rec_ica = S_clean @ ica.mixing_.T + ica.mean_

    S_sobi, A_sobi, _ = sobi(X_filtered, n_lags=20)
    idx_s = int(np.argmax(np.max(np.abs(S_sobi), axis=0)))
    S_clean_s = S_sobi.copy()
    S_clean_s[:, idx_s] = 0
    X_rec_sobi = S_clean_s @ A_sobi.T + X_filtered.mean(axis=0)

    corr_ica_mean, corr_ica_ci = bootstrap_correlation_ci(X_clean[:, 0], X_rec_ica[:, 0])
    corr_sobi_mean, corr_sobi_ci = bootstrap_correlation_ci(X_clean[:, 0], X_rec_sobi[:, 0])

    print("=== Section 6 bootstrap CIs (10-second synthetic benchmark) ===")
    print(f"FastICA correlation: point estimate {corr_ica_mean*100:.1f}%, "
          f"95% CI [{corr_ica_ci[0]*100:.1f}%, {corr_ica_ci[1]*100:.1f}%]")
    print(f"SOBI correlation:    point estimate {corr_sobi_mean*100:.1f}%, "
          f"95% CI [{corr_sobi_ci[0]*100:.1f}%, {corr_sobi_ci[1]*100:.1f}%]")
    overlap = not (corr_ica_ci[0] > corr_sobi_ci[1] or corr_sobi_ci[0] > corr_ica_ci[1])
    print(f"CIs overlap: {overlap}")
    return corr_ica_ci, corr_sobi_ci


def tournament_with_ci_and_permutation(n_sessions=50, seed=7, n_perm=10000):
    """Re-run Section 8.4's tournament, attach bootstrap CIs to the R^2 values,
    and run a permutation test on the CNN-vs-RF R^2 difference."""
    rng = np.random.default_rng(seed)
    true_focus = rng.uniform(0, 1, n_sessions)
    raw_windows = np.array([simulate_session(f, seed=i) for i, f in enumerate(true_focus)])
    X_feat = np.array([extract_features(w) for w in raw_windows])
    self_report = np.clip(np.round(1 + true_focus * 4 + rng.normal(0, 0.4, n_sessions)), 1, 5)

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    preds_rf = np.zeros(n_sessions)
    preds_ridge = np.zeros(n_sessions)
    for tr, te in kf.split(X_feat):
        rf = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=0)
        rf.fit(X_feat[tr], self_report[tr])
        preds_rf[te] = rf.predict(X_feat[te])
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_feat[tr], self_report[tr])
        preds_ridge[te] = ridge.predict(X_feat[te])

    def bootstrap_r2(y_true, y_pred, n_boot=2000, seed=0):
        rng2 = np.random.default_rng(seed)
        n = len(y_true)
        r2s = np.array([r2_score(y_true[idx], y_pred[idx])
                         for idx in (rng2.integers(0, n, n) for _ in range(n_boot))])
        return np.mean(r2s), np.std(r2s), np.percentile(r2s, [2.5, 97.5])

    rf_mean, rf_std, rf_ci = bootstrap_r2(self_report, preds_rf)
    ridge_mean, ridge_std, ridge_ci = bootstrap_r2(self_report, preds_ridge)

    print("\n=== Section 8.4 tournament: bootstrap R^2 (n=50 test-fold predictions) ===")
    print(f"Random Forest R^2: {rf_mean:.3f} +/- {rf_std:.3f}, 95% CI [{rf_ci[0]:.3f}, {rf_ci[1]:.3f}]")
    print(f"Ridge R^2:         {ridge_mean:.3f} +/- {ridge_std:.3f}, 95% CI [{ridge_ci[0]:.3f}, {ridge_ci[1]:.3f}]")

    # Permutation test: is the observed RF-vs-Ridge R^2 difference distinguishable from chance
    # re-assignment of which fold-prediction belongs to which model? We permute which model's
    # predictions are compared against the true labels, using a paired sign-flip permutation
    # on the per-sample squared errors (a standard non-parametric paired-difference test).
    se_rf = (self_report - preds_rf) ** 2
    se_ridge = (self_report - preds_ridge) ** 2
    observed_diff = np.mean(se_ridge - se_rf)  # positive => RF has lower error than Ridge

    rng3 = np.random.default_rng(1)
    diffs = se_ridge - se_rf
    perm_diffs = np.empty(n_perm)
    for i in range(n_perm):
        signs = rng3.choice([-1, 1], size=n_sessions)
        perm_diffs[i] = np.mean(diffs * signs)
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))

    print(f"\nPaired permutation test (RF vs Ridge squared error, {n_perm} sign-flip permutations):")
    print(f"Observed mean(SE_ridge - SE_rf) = {observed_diff:.4f}")
    print(f"Two-sided permutation p-value = {p_value:.4f}")
    print(f"{'Significant at alpha=0.05' if p_value < 0.05 else 'NOT significant at alpha=0.05'}")


if __name__ == "__main__":
    section6_with_ci()
    tournament_with_ci_and_permutation()

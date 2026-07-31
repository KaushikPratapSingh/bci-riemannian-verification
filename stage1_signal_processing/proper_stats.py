import sys
import time
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.decomposition import FastICA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from sobi import sobi  # noqa: E402
from phase4_tournament import simulate_session, extract_features  # noqa: E402
from mini_eegnet import MiniEEGNet  # noqa: E402


def run_one_benchmark(seed):
    """One full synthetic FastICA-vs-SOBI run at a given seed -- same generative
    model as Section 6, varied only by the random draw of background noise,
    blink-injection randomness being held fixed (the blink TIMING/amplitude is
    deterministic in the original script; only background noise varies here,
    which is itself the honest scope of what 'a different realization' means
    for this specific synthetic generator)."""
    rng_state = np.random.RandomState(seed)
    fs = 250
    t = np.arange(0, 10, 1 / fs)
    source_theta = np.sin(2 * np.pi * 6 * t) * 0.5
    source_alpha = np.sin(2 * np.pi * 10 * t) * 1.2
    source_beta = np.sin(2 * np.pi * 20 * t) * 0.8
    source_bg = rng_state.randn(len(t)) * 0.3
    S_neural = np.c_[source_theta, source_alpha, source_beta, source_bg]
    A_neural = np.array([[0.8, 0.2, 0.1, 0.1], [0.2, 0.8, 0.1, 0.1],
                          [0.1, 0.2, 0.8, 0.2], [0.1, 0.1, 0.2, 0.8]])
    X_clean = S_neural @ A_neural.T

    blink_signal = np.zeros(len(t))
    blink_signal[int(1.8 * fs):int(2.2 * fs)] = np.hanning(int(0.4 * fs)) * 15
    blink_signal[int(6.8 * fs):int(7.2 * fs)] = np.hanning(int(0.4 * fs)) * 15
    muscle_noise = np.zeros(len(t))
    muscle_noise[int(4 * fs):int(5 * fs)] = rng_state.randn(fs) * 5
    X_noisy = X_clean.copy()
    X_noisy[:, 0] += blink_signal + muscle_noise * 0.8
    X_noisy[:, 1] += blink_signal + muscle_noise * 0.8

    nyq = 0.5 * fs
    b, a = butter(4, [1.0 / nyq, 30.0 / nyq], btype='band')
    X_filtered = filtfilt(b, a, X_noisy, axis=0)

    ica = FastICA(n_components=4, random_state=seed)
    S_ica = ica.fit_transform(X_filtered)
    idx = int(np.argmax(np.max(np.abs(S_ica), axis=0)))
    S_clean = S_ica.copy()
    S_clean[:, idx] = 0
    X_rec_ica = S_clean @ ica.mixing_.T + ica.mean_
    corr_ica = np.corrcoef(X_clean[:, 0], X_rec_ica[:, 0])[0, 1]

    S_sobi, A_sobi, _ = sobi(X_filtered, n_lags=20)
    idx_s = int(np.argmax(np.max(np.abs(S_sobi), axis=0)))
    S_clean_s = S_sobi.copy()
    S_clean_s[:, idx_s] = 0
    X_rec_sobi = S_clean_s @ A_sobi.T + X_filtered.mean(axis=0)
    corr_sobi = np.corrcoef(X_clean[:, 0], X_rec_sobi[:, 0])[0, 1]

    return corr_ica, corr_sobi


def between_recording_variability(n_runs=200):
    corr_icas, corr_sobis = [], []
    for seed in range(n_runs):
        ci, cs = run_one_benchmark(seed)
        corr_icas.append(ci)
        corr_sobis.append(cs)
    corr_icas, corr_sobis = np.array(corr_icas), np.array(corr_sobis)
    print(f"=== Between-recording variability across {n_runs} independent synthetic realizations ===")
    print(f"FastICA: mean={np.mean(corr_icas)*100:.1f}%, std={np.std(corr_icas)*100:.1f}%, "
          f"95% range=[{np.percentile(corr_icas,2.5)*100:.1f}%, {np.percentile(corr_icas,97.5)*100:.1f}%]")
    print(f"SOBI:    mean={np.mean(corr_sobis)*100:.1f}%, std={np.std(corr_sobis)*100:.1f}%, "
          f"95% range=[{np.percentile(corr_sobis,2.5)*100:.1f}%, {np.percentile(corr_sobis,97.5)*100:.1f}%]")
    diff = corr_icas - corr_sobis
    print(f"FastICA-SOBI difference: mean={np.mean(diff)*100:.1f} pts, "
          f"fraction of {n_runs} runs where FastICA > SOBI: {np.mean(diff>0)*100:.0f}%")
    return corr_icas, corr_sobis


def cnn_vs_rf_real_comparison(n_sessions=50, seed=7, n_boot=2000):
    """The specific comparison the review flagged: get real frozen out-of-fold
    predictions for CNN and RF once, then bootstrap/permute on those -- not by
    retraining the CNN thousands of times."""
    rng = np.random.default_rng(seed)
    true_focus = rng.uniform(0, 1, n_sessions)
    raw_windows = np.array([simulate_session(f, seed=i) for i, f in enumerate(true_focus)])
    X_feat = np.array([extract_features(w) for w in raw_windows])
    self_report = np.clip(np.round(1 + true_focus * 4 + rng.normal(0, 0.4, n_sessions)), 1, 5)

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    preds_rf = np.zeros(n_sessions)
    preds_cnn = np.zeros(n_sessions)

    t0 = time.time()
    for tr, te in kf.split(X_feat):
        rf = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=0)
        rf.fit(X_feat[tr], self_report[tr])
        preds_rf[te] = rf.predict(X_feat[te])

        net = MiniEEGNet(c_in=4, n_filters=8, kernel_size=25, seed=0)
        y_mean, y_std = self_report[tr].mean(), self_report[tr].std() + 1e-8
        y_norm = (self_report[tr] - y_mean) / y_std
        for epoch in range(60):
            order = np.random.default_rng(epoch).permutation(len(tr))
            for idx in order:
                net.train_step(raw_windows[tr][idx], y_norm[idx], 0.01)
        preds_cnn[te] = np.array([net.forward(x) for x in raw_windows[te]]) * y_std + y_mean
    print(f"(real CNN+RF training across 5 folds took {time.time()-t0:.1f}s)\n")

    r2_rf = r2_score(self_report, preds_rf)
    r2_cnn = r2_score(self_report, preds_cnn)
    print(f"Frozen predictions -- Random Forest R^2={r2_rf:.3f}, CNN R^2={r2_cnn:.3f}, diff={r2_cnn-r2_rf:.3f}")

    # Bootstrap CI on R^2 for each, using the SAME frozen predictions (fast, no retraining)
    def boot_r2(y_true, y_pred, n_boot=n_boot, seed=0):
        rng2 = np.random.default_rng(seed)
        n = len(y_true)
        r2s = np.array([r2_score(y_true[idx], y_pred[idx])
                         for idx in (rng2.integers(0, n, n) for _ in range(n_boot))])
        return np.mean(r2s), np.percentile(r2s, [2.5, 97.5])

    rf_mean, rf_ci = boot_r2(self_report, preds_rf)
    cnn_mean, cnn_ci = boot_r2(self_report, preds_cnn)
    print(f"RF  bootstrap: {rf_mean:.3f}, 95% CI [{rf_ci[0]:.3f}, {rf_ci[1]:.3f}]")
    print(f"CNN bootstrap: {cnn_mean:.3f}, 95% CI [{cnn_ci[0]:.3f}, {cnn_ci[1]:.3f}]")
    overlap = not (rf_ci[0] > cnn_ci[1] or cnn_ci[0] > rf_ci[1])
    print(f"CIs overlap: {overlap}")

    # Paired permutation test (sign-flip on per-sample squared-error difference)
    se_rf = (self_report - preds_rf) ** 2
    se_cnn = (self_report - preds_cnn) ** 2
    diffs = se_rf - se_cnn  # positive => CNN has lower error than RF
    observed = np.mean(diffs)
    rng3 = np.random.default_rng(2)
    n_perm = 10000
    perm = np.array([np.mean(diffs * rng3.choice([-1, 1], size=n_sessions)) for _ in range(n_perm)])
    p_value = np.mean(np.abs(perm) >= np.abs(observed))
    print(f"\nPaired permutation test (CNN vs RF squared error, {n_perm} permutations):")
    print(f"Observed mean(SE_rf - SE_cnn) = {observed:.4f}")
    print(f"Two-sided p-value = {p_value:.4f} -> "
          f"{'significant' if p_value < 0.05 else 'NOT significant'} at alpha=0.05")


if __name__ == "__main__":
    between_recording_variability(n_runs=200)
    print()
    cnn_vs_rf_real_comparison()

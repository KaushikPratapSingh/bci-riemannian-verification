"""
Phase IV -- Classical ML vs. Deep Learning BCI Tournament
=============================================================
1. Reuses simulate_session() from this project's stage_a_pipeline_demo.py
2. extract_features() builds the 18-dim feature vector (12 band-power + 6 connectivity)
3. MiniEEGNet (mini_eegnet.py) -- a from-scratch, gradient-checked 1D CNN on raw voltages
4. RandomForestRegressor / Ridge on the engineered features
5. 5-fold CV under both intact and shuffled (negative control) labels
6. A formatted comparison table: R^2, Pearson r, training time, inference latency

This is an independent, small-scale replication of the question Paredes
Ocaranza et al. (2025) [12] answered at larger scale: does a deep network
beat classical ML on low-channel, small-sample consumer EEG, or does it
overfit? Run honestly, on this project's own synthetic pipeline, with a
negative control gate identical in spirit to theirs.
"""

import sys
import time
import numpy as np
from scipy.signal import welch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

sys.path.insert(0, "/home/claude/storm_phases")
from mini_eegnet import MiniEEGNet, numerical_gradient_check  # noqa: E402

FS = 250
BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}


def bandpower(sig, fs, band):
    f, pxx = welch(sig, fs=fs, nperseg=fs * 2)
    mask = (f >= band[0]) & (f <= band[1])
    return np.trapezoid(pxx[mask], f[mask])


def extract_features(window_4ch, fs=FS):
    feats = []
    for ch in range(4):
        for band in BANDS.values():
            feats.append(bandpower(window_4ch[:, ch], fs, band))
    for i in range(4):
        for j in range(i + 1, 4):
            feats.append(np.corrcoef(window_4ch[:, i], window_4ch[:, j])[0, 1])
    return np.array(feats)


def simulate_session(true_focus, fs=FS, duration_s=4, seed=None):
    """Same generative model as stage_a_pipeline_demo.py, shortened to 4s
    (instead of 10s) purely so the from-scratch CNN trains in a reasonable
    time on CPU in this sandbox -- the underlying synthetic structure
    (and its honesty caveats) is identical."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration_s, 1 / fs)
    theta = np.sin(2 * np.pi * 6 * t) * (0.8 - 0.4 * true_focus)
    alpha = np.sin(2 * np.pi * 10 * t) * (1.0 - 0.3 * true_focus)
    beta = np.sin(2 * np.pi * 20 * t) * (0.4 + 0.9 * true_focus)
    bg = rng.standard_normal(len(t)) * 0.3
    S = np.c_[theta, alpha, beta, bg]
    A = np.array([[0.8, 0.2, 0.1, 0.1], [0.2, 0.8, 0.1, 0.1],
                  [0.1, 0.2, 0.8, 0.2], [0.1, 0.1, 0.2, 0.8]])
    X = S @ A.T + rng.standard_normal((len(t), 4)) * 0.15
    return X


def train_cnn(X_train_raw, y_train, n_epochs=60, lr=0.01, seed=0):
    net = MiniEEGNet(c_in=4, n_filters=8, kernel_size=25, seed=seed)
    y_mean, y_std = np.mean(y_train), np.std(y_train) + 1e-8
    y_train_norm = (y_train - y_mean) / y_std
    for epoch in range(n_epochs):
        order = np.random.default_rng(epoch).permutation(len(X_train_raw))
        for idx in order:
            net.train_step(X_train_raw[idx], y_train_norm[idx], lr)
    return net, y_mean, y_std


def predict_cnn(net, X_raw, y_mean, y_std):
    return np.array([net.forward(x) for x in X_raw]) * y_std + y_mean


def run_fold_classical(model_cls, X_train, y_train, X_test, **kwargs):
    t0 = time.perf_counter()
    model = model_cls(**kwargs)
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    preds = model.predict(X_test)
    infer_time = (time.perf_counter() - t0) / len(X_test)
    return preds, train_time, infer_time


def run_fold_cnn(X_train_raw, y_train, X_test_raw):
    t0 = time.perf_counter()
    net, y_mean, y_std = train_cnn(X_train_raw, y_train)
    train_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    preds = predict_cnn(net, X_test_raw, y_mean, y_std)
    infer_time = (time.perf_counter() - t0) / len(X_test_raw)
    return preds, train_time, infer_time


def evaluate(n_sessions=50, seed=7):
    print("Step 0: mandatory gradient check before trusting the CNN on any data...")
    if not numerical_gradient_check():
        raise SystemExit("Gradient check failed.")
    print()

    rng = np.random.default_rng(seed)
    true_focus = rng.uniform(0, 1, n_sessions)
    raw_windows = np.array([simulate_session(f, seed=i) for i, f in enumerate(true_focus)])
    X_feat = np.array([extract_features(w) for w in raw_windows])
    self_report = np.clip(np.round(1 + true_focus * 4 + rng.normal(0, 0.4, n_sessions)), 1, 5)
    shuffled_report = rng.permutation(self_report)

    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    results = {}
    for label_name, labels in [("intact", self_report), ("shuffled (negative control)", shuffled_report)]:
        for model_name in ["Random Forest", "Ridge", "MiniEEGNet (from scratch CNN)"]:
            preds = np.zeros(n_sessions)
            train_times, infer_times = [], []
            for train_idx, test_idx in kf.split(X_feat):
                if model_name == "Random Forest":
                    p, tt, it = run_fold_classical(RandomForestRegressor, X_feat[train_idx],
                                                    labels[train_idx], X_feat[test_idx],
                                                    n_estimators=100, max_depth=4, random_state=0)
                elif model_name == "Ridge":
                    p, tt, it = run_fold_classical(Ridge, X_feat[train_idx], labels[train_idx],
                                                    X_feat[test_idx], alpha=1.0)
                else:
                    p, tt, it = run_fold_cnn(raw_windows[train_idx], labels[train_idx], raw_windows[test_idx])
                preds[test_idx] = p
                train_times.append(tt)
                infer_times.append(it)
            corr = np.corrcoef(labels, preds)[0, 1]
            r2 = r2_score(labels, preds)
            results[(label_name, model_name)] = {
                "r2": r2, "corr": corr,
                "train_s": np.mean(train_times), "infer_ms": np.mean(infer_times) * 1000,
            }

    print(f"{'Labels':<28}{'Model':<32}{'R^2':>8}{'Pearson r':>12}{'Train (s)':>12}{'Infer (ms)':>12}")
    print("-" * 104)
    for (label_name, model_name), m in results.items():
        print(f"{label_name:<28}{model_name:<32}{m['r2']:>8.3f}{m['corr']:>12.3f}"
              f"{m['train_s']:>12.4f}{m['infer_ms']:>12.4f}")

    print("\nValidation gate: under shuffled labels, both models should show r ~ 0 and R^2 < 0.")
    for model_name in ["Random Forest", "Ridge", "MiniEEGNet (from scratch CNN)"]:
        m = results[("shuffled (negative control)", model_name)]
        gate = "PASS" if m["r2"] < 0.0 else "FAIL (overfitting to noise)"
        print(f"  {model_name:<32} R^2={m['r2']:.3f}  -> {gate}")


if __name__ == "__main__":
    evaluate()

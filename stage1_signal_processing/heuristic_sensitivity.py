import sys
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.stats import kurtosis
from sklearn.decomposition import FastICA
from sobi import sobi  # noqa: E402


def build_dataset(seed=42):
    rng = np.random.RandomState(seed)
    fs = 250
    t = np.arange(0, 10, 1 / fs)
    source_theta = np.sin(2 * np.pi * 6 * t) * 0.5
    source_alpha = np.sin(2 * np.pi * 10 * t) * 1.2
    source_beta = np.sin(2 * np.pi * 20 * t) * 0.8
    source_bg = rng.randn(len(t)) * 0.3
    S = np.c_[source_theta, source_alpha, source_beta, source_bg]
    A = np.array([[0.8, 0.2, 0.1, 0.1], [0.2, 0.8, 0.1, 0.1],
                  [0.1, 0.2, 0.8, 0.2], [0.1, 0.1, 0.2, 0.8]])
    X_clean = S @ A.T
    blink = np.zeros(len(t))
    blink[int(1.8 * fs):int(2.2 * fs)] = np.hanning(int(0.4 * fs)) * 15
    blink[int(6.8 * fs):int(7.2 * fs)] = np.hanning(int(0.4 * fs)) * 15
    muscle = np.zeros(len(t))
    muscle[int(4 * fs):int(5 * fs)] = rng.randn(fs) * 5
    X_noisy = X_clean.copy()
    X_noisy[:, 0] += blink + muscle * 0.8
    X_noisy[:, 1] += blink + muscle * 0.8
    nyq = 0.5 * fs
    b, a = butter(4, [1.0 / nyq, 30.0 / nyq], btype='band')
    X_filtered = filtfilt(b, a, X_noisy, axis=0)
    return X_clean, X_filtered


def evaluate(X_clean, X_filtered, S, A, mean, heuristic):
    if heuristic == "max_amplitude":
        idx = int(np.argmax(np.max(np.abs(S), axis=0)))
    elif heuristic == "max_kurtosis":
        idx = int(np.argmax(np.abs(kurtosis(S, axis=0, fisher=True))))
    else:
        raise ValueError(heuristic)
    S_clean = S.copy()
    S_clean[:, idx] = 0
    X_rec = S_clean @ A.T + mean
    corr = np.corrcoef(X_clean[:, 0], X_rec[:, 0])[0, 1]
    return idx, corr


def main():
    X_clean, X_filtered = build_dataset()

    ica = FastICA(n_components=4, random_state=42, max_iter=2000)
    S_ica = ica.fit_transform(X_filtered)
    A_ica, mean_ica = ica.mixing_, ica.mean_

    S_sobi, A_sobi, _ = sobi(X_filtered, n_lags=20)
    mean_sobi = X_filtered.mean(axis=0)

    print(f"{'Algorithm':<10}{'Heuristic':<16}{'Component idx':>14}{'Correlation':>13}")
    print("-" * 53)
    for name, S, A, mean in [("FastICA", S_ica, A_ica, mean_ica), ("SOBI", S_sobi, A_sobi, mean_sobi)]:
        for heuristic in ["max_amplitude", "max_kurtosis"]:
            idx, corr = evaluate(X_clean, X_filtered, S, A, mean, heuristic)
            print(f"{name:<10}{heuristic:<16}{idx:>14}{corr*100:>12.1f}%")


if __name__ == "__main__":
    main()

"""
A second, more realistic synthetic benchmark: AR(4)-filtered noise sources
instead of pure sine tones, addressing the review's point that pure
sinusoids are an idealized signal model real EEG does not match.
"""

import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.decomposition import FastICA
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sobi import sobi  # noqa: E402


def ar4_source(n_samples, a_coeffs, rng, drive_std=1.0):
    """AR(4) process: x[t] = sum(a_coeffs * x[t-4:t]) + noise. A standard
    stochastic model for band-limited EEG-like activity, in contrast to a
    deterministic sine tone."""
    x = np.zeros(n_samples)
    drive = rng.standard_normal(n_samples) * drive_std
    for t in range(4, n_samples):
        x[t] = np.dot(a_coeffs, x[t - 4:t][::-1]) + drive[t]
    return x


def design_ar4_for_peak(target_freq_hz, fs, bandwidth=2.0):
    """Designs AR(4) coefficients with a spectral peak near target_freq_hz by
    using two complex-conjugate pole pairs placed at that frequency, then
    converting to AR coefficients via the characteristic polynomial. This is
    a standard, simple way to get a resonant AR model -- not Burg's method,
    but transparent and checkable."""
    r = 1 - (bandwidth / fs) * np.pi  # pole radius (closer to 1 = narrower peak)
    theta = 2 * np.pi * target_freq_hz / fs
    pole = r * np.exp(1j * theta)
    poles = [pole, np.conj(pole), pole, np.conj(pole)]  # double pair for AR(4)
    char_poly = np.poly(poles)  # coefficients of (z-p1)(z-p2)(z-p3)(z-p4)
    a_coeffs = -char_poly[1:5].real  # AR coefficients (x[t] = sum a_k x[t-k] + e[t])
    return a_coeffs


def run_ar4_benchmark(seed=42):
    rng = np.random.default_rng(seed)
    fs = 250
    n = 10 * fs

    a_theta = design_ar4_for_peak(6, fs)
    a_alpha = design_ar4_for_peak(10, fs)
    a_beta = design_ar4_for_peak(20, fs)

    source_theta = ar4_source(n, a_theta, rng)
    source_alpha = ar4_source(n, a_alpha, rng)
    source_beta = ar4_source(n, a_beta, rng)
    source_bg = rng.standard_normal(n) * 0.3

    # normalize each source to unit variance before scaling, so the AR
    # filter's own gain doesn't silently change the relative source amplitudes
    for s in (source_theta, source_alpha, source_beta):
        s /= (np.std(s) + 1e-9)
    source_theta = source_theta * 0.5
    source_alpha = source_alpha * 1.2
    source_beta = source_beta * 0.8

    S_neural = np.c_[source_theta, source_alpha, source_beta, source_bg]
    A_neural = np.array([[0.8, 0.2, 0.1, 0.1], [0.2, 0.8, 0.1, 0.1],
                          [0.1, 0.2, 0.8, 0.2], [0.1, 0.1, 0.2, 0.8]])
    X_clean = S_neural @ A_neural.T

    blink_signal = np.zeros(n)
    blink_signal[int(1.8 * fs):int(2.2 * fs)] = np.hanning(int(0.4 * fs)) * 15
    blink_signal[int(6.8 * fs):int(7.2 * fs)] = np.hanning(int(0.4 * fs)) * 15
    muscle_noise = np.zeros(n)
    muscle_noise[int(4 * fs):int(5 * fs)] = rng.standard_normal(fs) * 5
    X_noisy = X_clean.copy()
    X_noisy[:, 0] += blink_signal + muscle_noise * 0.8
    X_noisy[:, 1] += blink_signal + muscle_noise * 0.8

    nyq = 0.5 * fs
    b, a = butter(4, [1.0 / nyq, 30.0 / nyq], btype='band')
    X_filtered = filtfilt(b, a, X_noisy, axis=0)

    def snr(clean, test):
        noise = test - clean
        return 10 * np.log10(np.sum(clean ** 2) / np.sum(noise ** 2))

    snr_before = snr(X_clean[:, 0], X_noisy[:, 0])
    corr_before = np.corrcoef(X_clean[:, 0], X_noisy[:, 0])[0, 1]

    ica = FastICA(n_components=4, random_state=42, max_iter=2000, tol=1e-4)
    S_ica = ica.fit_transform(X_filtered)
    idx = int(np.argmax(np.max(np.abs(S_ica), axis=0)))
    S_clean = S_ica.copy()
    S_clean[:, idx] = 0
    X_rec_ica = S_clean @ ica.mixing_.T + ica.mean_
    snr_ica = snr(X_clean[:, 0], X_rec_ica[:, 0])
    corr_ica = np.corrcoef(X_clean[:, 0], X_rec_ica[:, 0])[0, 1]

    S_sobi, A_sobi, _ = sobi(X_filtered, n_lags=20)
    idx_s = int(np.argmax(np.max(np.abs(S_sobi), axis=0)))
    S_clean_s = S_sobi.copy()
    S_clean_s[:, idx_s] = 0
    X_rec_sobi = S_clean_s @ A_sobi.T + X_filtered.mean(axis=0)
    snr_sobi = snr(X_clean[:, 0], X_rec_sobi[:, 0])
    corr_sobi = np.corrcoef(X_clean[:, 0], X_rec_sobi[:, 0])[0, 1]

    print("=== AR(4) stochastic benchmark (vs. Section 6's pure-sine benchmark) ===")
    print(f"{'Metric':<28}{'Before':>10}{'FastICA':>10}{'SOBI':>10}")
    print(f"{'SNR (dB)':<28}{snr_before:>10.2f}{snr_ica:>10.2f}{snr_sobi:>10.2f}")
    print(f"{'Correlation':<28}{corr_before*100:>9.1f}%{corr_ica*100:>9.1f}%{corr_sobi*100:>9.1f}%")
    return corr_ica, corr_sobi


if __name__ == "__main__":
    run_ar4_benchmark()

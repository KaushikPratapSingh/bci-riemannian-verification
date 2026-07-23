"""
Phase I diagnostic: why did both gates fail on real PhysioNet data?

Tests a specific hypothesis: that "clean" means something different for
real data than for synthetic data. On the synthetic benchmark, the clean
reference is a true noise-free signal by construction. On real PhysioNet
data, the "clean" reference is just the raw recording -- it already
carries its own drift, alpha rhythm, and possibly its own blinks, with no
noise-free version available to compare against.

This script filters the reference signal the same way (50 Hz notch, then
1-30 Hz bandpass) that the recovered signal is filtered, before computing
SNR and correlation -- instead of comparing against the fully raw
reference, which is what the original validation script does. Neither the
pre-declared gate thresholds nor the artifact injection amplitudes are
changed; only the reference signal used for comparison is.

Requires sobi.py in the same folder, and the PhysioNet data already
downloaded once by phase1_physionet_validation.py (mne caches it locally
after the first download, so this does not need network access to run
again).
"""

import sys
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.decomposition import FastICA
import scipy.stats as stats

try:
    import mne
    from mne.datasets import eegbci
    from mne.io import read_raw_edf
except ImportError:
    print("Error: 'mne' is required. Run: pip install mne")
    sys.exit(1)

try:
    from sobi import sobi
except ImportError:
    print("Error: place sobi.py in the same folder as this script.")
    sys.exit(1)


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data, axis=0)


def notch_filter(data, notch_freq, fs, q=30.0):
    nyq = 0.5 * fs
    w0 = notch_freq / nyq
    bw = w0 / q
    b, a = butter(2, [w0 - bw / 2, w0 + bw / 2], btype="bandstop")
    return filtfilt(b, a, data, axis=0)


def snr(clean, test):
    noise = test - clean
    return 10 * np.log10(np.sum(clean ** 2) / np.sum(noise ** 2))


def main():
    print("\nRunning Phase I diagnostic on PhysioNet data...\n")

    edf_files = eegbci.load_data(subjects=[1], runs=[1])
    raw = read_raw_edf(edf_files[0], preload=True, verbose=False)
    fs = int(raw.info["sfreq"])

    target_channels = ["Fp1.", "Fp2.", "F3..", "F4.."]
    raw_frontal = raw.copy().pick(target_channels)
    data = raw_frontal.get_data() * 1e6  # V -> uV

    X_clean_raw = data[:, : 10 * fs].T  # 10-second window, (samples, channels)

    raw_rms = np.sqrt(np.mean(X_clean_raw[:, 0] ** 2))
    print(f"Raw Fp1 channel RMS amplitude: {raw_rms:.2f} uV")
    print("Injected artifacts: 35.0 uV peak / 15.0 uV RMS.")
    print("A real channel's own amplitude can dwarf a fixed-microvolt artifact")
    print("injection that was calibrated against a much smaller synthetic source.\n")

    n_samples = len(X_clean_raw)
    blink_signal = np.zeros(n_samples)
    blink_times = np.linspace(1.5, 8.5, 6)
    for bt in blink_times:
        idx = int(bt * fs)
        half = int(0.25 * fs)
        if idx - half > 0 and idx + half < n_samples:
            blink_signal[idx - half : idx + half] = np.hanning(2 * half) * 35.0

    rng = np.random.default_rng(0)
    muscle_noise = np.zeros(n_samples)
    emg_s, emg_e = int(4.0 * fs), int(5.0 * fs)
    muscle_noise[emg_s:emg_e] = rng.standard_normal(emg_e - emg_s) * 15.0

    X_noisy = X_clean_raw.copy()
    X_noisy[:, 0] += blink_signal + muscle_noise * 0.8
    X_noisy[:, 1] += blink_signal + muscle_noise * 0.8

    X_notched = notch_filter(X_noisy, 50.0, fs)
    X_filtered = butter_bandpass_filter(X_notched, 1.0, 30.0, fs)

    ica = FastICA(n_components=4, random_state=42, max_iter=2000)
    S_recon_ica = ica.fit_transform(X_filtered)
    blink_idx_ica = np.argmax(np.max(np.abs(S_recon_ica), axis=0))
    S_cleaned_ica = S_recon_ica.copy()
    S_cleaned_ica[:, blink_idx_ica] = 0
    X_rec = np.dot(S_cleaned_ica, ica.mixing_.T) + ica.mean_

    snr_before_orig = snr(X_clean_raw[:, 0], X_noisy[:, 0])
    corr_after_orig, _ = stats.pearsonr(X_clean_raw[:, 0], X_rec[:, 0])

    print("Original evaluation (recovered vs. unfiltered raw reference):")
    print(f"  Pre-pipeline SNR:            {snr_before_orig:+.2f} dB  (gate: -20 to -15)")
    print(f"  Post-pipeline correlation:    {corr_after_orig*100:.1f}%   (gate: >= 40.0%)")

    X_clean_filtered = butter_bandpass_filter(notch_filter(X_clean_raw, 50.0, fs), 1.0, 30.0, fs)
    snr_before_corr = snr(X_clean_filtered[:, 0], X_noisy[:, 0])
    corr_after_corr, _ = stats.pearsonr(X_clean_filtered[:, 0], X_rec[:, 0])

    print("\nCorrected evaluation (recovered vs. similarly-filtered reference):")
    print(f"  Pre-pipeline SNR:             {snr_before_corr:+.2f} dB")
    print(f"  Post-pipeline correlation:    {corr_after_corr*100:.1f}%")
    print("\nNeither the gate thresholds above nor the injection amplitudes were")
    print("changed between the two evaluations -- only the reference signal used")
    print("for comparison. Report both sets of numbers, not only the more")
    print("favorable one.")


if __name__ == "__main__":
    main()

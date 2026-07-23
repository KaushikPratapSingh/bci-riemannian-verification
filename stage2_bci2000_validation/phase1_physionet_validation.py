"""
Phase I -- Real-World PhysioNet Empirical Validation
========================================================
*** THIS SCRIPT HAS NOT BEEN RUN. ***
This sandbox has no network access, so it cannot download PhysioNet data.
Every other script in this project (Sections 6, 8, and STORM Phases II-IV)
was actually executed and its real output reported. This one could not be,
and the honest thing to do is say so plainly rather than fabricate numbers
that look plausible -- which is the exact failure mode this roadmap itself
warns against.

Run this on a machine with internet access:
    pip install mne numpy scipy scikit-learn matplotlib
    python3 phase1_physionet_validation.py

Then report the actual printed output back. It will be checked against the
expected bounds in the validation gate at the bottom of this docstring
before being written into the paper as a real result -- not before.

Expected bounds (from the STORM roadmap's own pre-declared gate, consistent
with this project's Section 6.3 synthetic benchmark):
  - raw.info['sfreq'] must print as 160.0 (PhysioNet's BCI2000 native rate,
    NOT this project's own 250 SPS hardware design -- if this prints
    anything else, something in the loader is wrong)
  - Pre-pipeline SNR should fall roughly between -15 dB and -20 dB
  - Post-pipeline correlation should reach at least r = 0.40
"""

import sys
import numpy as np
from scipy.signal import butter, filtfilt
import scipy.stats as stats

sys.path.insert(0, "/home/claude/paper")  # this project's verified sobi.py

try:
    import mne
    from mne.datasets import eegbci
    from mne.io import read_raw_edf
except ImportError:
    print("This script requires mne: pip install mne numpy scipy scikit-learn matplotlib")
    sys.exit(1)

from sklearn.decomposition import FastICA
from sobi import sobi  # noqa: E402


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, data, axis=0)


def notch_filter(data, notch_freq, fs, q=30.0):
    nyq = 0.5 * fs
    w0 = notch_freq / nyq
    bw = w0 / q
    b, a = butter(2, [w0 - bw / 2, w0 + bw / 2], btype="bandstop")
    return filtfilt(b, a, data, axis=0)


def calculate_snr(clean_sig, test_sig):
    noise = test_sig - clean_sig
    return 10 * np.log10(np.sum(clean_sig ** 2) / np.sum(noise ** 2))


def main():
    # STEP 1: download and load real human resting-state EEG
    print("[1/6] Downloading PhysioNet BCI2000 Subject 1, Run 1 (eyes-open resting baseline)...")
    edf_files = eegbci.load_data(subjects=[1], runs=[1])
    raw = read_raw_edf(edf_files[0], preload=True, verbose=False)
    fs = raw.info["sfreq"]
    print(f"      VERIFIED metadata: raw.info['sfreq'] = {fs}  "
          f"(expected: 160.0 -- PhysioNet's native rate, not this project's 250 SPS)")
    print(f"      Channels available: {len(raw.info['ch_names'])}")

    # STEP 2: extract the frontal 4-channel subset matching this project's electrode array
    target = [ch for ch in raw.info["ch_names"] if ch.strip(".").upper() in
              ("FP1", "FP2", "F3", "F4")]
    print(f"      Frontal channels matched in this file: {target}")
    if len(target) != 4:
        print("      WARNING: did not find exactly 4 matching frontal channels -- "
              "print raw.info['ch_names'] in full and adjust the match list above "
              "before trusting anything downstream.")
    raw_frontal = raw.copy().pick(target)
    X_clean = raw_frontal.get_data().T * 1e6  # convert V -> uV, shape (n_samples, 4)

    duration_s = 10
    n_samples = int(duration_s * fs)
    X_clean = X_clean[:n_samples]
    t = np.arange(n_samples) / fs

    # STEP 3: inject the same style of stress-test artifacts used in Section 6
    print("[2/6] Injecting blink and EMG artifacts onto real EEG...")
    blink_signal = np.zeros(n_samples)
    blink_times = np.linspace(1.5, duration_s - 1.5, 6)
    for bt in blink_times:
        idx = int(bt * fs)
        half = int(0.2 * fs)
        if idx - half > 0 and idx + half < n_samples:
            blink_signal[idx - half: idx + half] = np.hanning(2 * half) * 35

    muscle_noise = np.zeros(n_samples)
    emg_start, emg_end = int(4.0 * fs), int(5.0 * fs)
    muscle_noise[emg_start:emg_end] = np.random.default_rng(0).standard_normal(emg_end - emg_start) * 15

    X_noisy = X_clean.copy()
    X_noisy[:, 0] += blink_signal + muscle_noise * 0.8
    X_noisy[:, 1] += blink_signal + muscle_noise * 0.8

    # STEP 4: filter
    print("[3/6] Applying 50 Hz notch and 1-30 Hz bandpass filters...")
    X_notched = notch_filter(X_noisy, 50.0, fs)
    X_filtered = butter_bandpass_filter(X_notched, 1.0, 30.0, fs)

    # STEP 5: run FastICA and SOBI side by side, exactly as in Section 6.3
    print("[4/6] Running FastICA...")
    ica = FastICA(n_components=4, random_state=42, max_iter=2000)
    S_ica = ica.fit_transform(X_filtered)
    idx_ica = int(np.argmax(np.max(np.abs(S_ica), axis=0)))
    S_ica_clean = S_ica.copy()
    S_ica_clean[:, idx_ica] = 0
    X_rec_ica = S_ica_clean @ ica.mixing_.T + ica.mean_

    print("[5/6] Running SOBI...")
    S_sobi, A_sobi, _ = sobi(X_filtered, n_lags=20)
    idx_sobi = int(np.argmax(np.max(np.abs(S_sobi), axis=0)))
    S_sobi_clean = S_sobi.copy()
    S_sobi_clean[:, idx_sobi] = 0
    X_rec_sobi = S_sobi_clean @ A_sobi.T + X_filtered.mean(axis=0)

    # STEP 6: metrics
    print("[6/6] Computing metrics on the Fp1 channel...\n")
    snr_before = calculate_snr(X_clean[:, 0], X_noisy[:, 0])
    snr_ica = calculate_snr(X_clean[:, 0], X_rec_ica[:, 0])
    snr_sobi = calculate_snr(X_clean[:, 0], X_rec_sobi[:, 0])
    corr_before, _ = stats.pearsonr(X_clean[:, 0], X_noisy[:, 0])
    corr_ica, _ = stats.pearsonr(X_clean[:, 0], X_rec_ica[:, 0])
    corr_sobi, _ = stats.pearsonr(X_clean[:, 0], X_rec_sobi[:, 0])

    print(f"{'Metric':<30}{'Before':>12}{'FastICA':>12}{'SOBI':>12}")
    print(f"{'SNR (dB)':<30}{snr_before:>12.2f}{snr_ica:>12.2f}{snr_sobi:>12.2f}")
    print(f"{'Correlation':<30}{corr_before*100:>11.1f}%{corr_ica*100:>11.1f}%{corr_sobi*100:>11.1f}%")

    print("\n--- Validation gate checks (per the STORM roadmap, Phase I, Section 4) ---")
    gate1 = -20.0 <= snr_before <= -15.0
    gate2 = corr_ica >= 0.40
    print(f"  Pre-pipeline SNR in [-20, -15] dB: {snr_before:.2f} dB -> {'PASS' if gate1 else 'FAIL'}")
    print(f"  Post-pipeline (FastICA) correlation >= 0.40: {corr_ica:.3f} -> {'PASS' if gate2 else 'FAIL'}")
    print("\nReport this entire console output back verbatim -- including any FAILs -- "
          "so it can be written into the paper exactly as it actually ran.")


if __name__ == "__main__":
    main()


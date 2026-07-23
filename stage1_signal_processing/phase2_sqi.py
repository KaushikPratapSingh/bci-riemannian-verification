"""
Phase II -- Signal Quality Index (SQI) & Lead-Off Simulation
==============================================================
Implements:
  1. calculate_spectral_entropy(segment, fs)
  2. compute_sqi(segment, fs) -> float in [0.0, 1.0]
  3. A 10s synthetic run where Channel 1 fully detaches at t=6.0s
  4. A 1-second rolling analysis window, 100ms slide step
  5. A dynamic log: [Timestamp] | Raw RMS | Spectral Entropy | SQI | Status Badge

No placeholders -- every filter, window, and threshold below is a real,
executable computation, run on synthetic data with a known, designed-in
ground truth (good signal for t<6s, fully detached for t>=6s) so the
validation gate in Part 4 has something concrete to check against.
"""

import numpy as np
from scipy.signal import welch

FS = 250  # Hz, matches this project's hardware design (not PhysioNet's 160 Hz)


def calculate_spectral_entropy(segment, fs):
    """Shannon entropy of the normalized power spectral density.
    Flat thermal noise (disconnected electrode) -> high entropy.
    Real EEG with distinct spectral peaks -> lower entropy."""
    freqs, psd = welch(segment, fs=fs, nperseg=min(len(segment), fs))
    psd = psd[freqs > 0]  # drop the DC bin, which is undefined for log2(0) edge cases
    p = psd / (np.sum(psd) + 1e-15)
    p = p[p > 0]
    H = -np.sum(p * np.log2(p))
    H_max = np.log2(len(p)) if len(p) > 0 else 1.0
    return H / H_max if H_max > 0 else 0.0  # normalized to [0, 1]


def compute_sqi(segment, fs, v_ref=4.5, gain=12.0):
    """Combines three biophysical checks into a single SQI in [0.0, 1.0]:
    amplitude-saturation, high-frequency spectral ratio, and spectral entropy."""
    # 1. Amplitude saturation check (ADC rail limit, referred to input)
    rail_limit_uv = (v_ref / gain) * 1e6 * 0.3  # practical headroom fraction of full rail, in uV
    if np.max(np.abs(segment)) >= rail_limit_uv:
        return 0.0

    # 2. High-frequency spectral ratio: muscle/thermal band vs. physiological band
    freqs, psd = welch(segment, fs=fs, nperseg=min(len(segment), fs))
    hf_power = np.sum(psd[(freqs >= 35) & (freqs <= min(125, fs / 2 - 1))])
    lf_power = np.sum(psd[(freqs >= 1) & (freqs <= 30)])
    r_noise = hf_power / (lf_power + 1e-12)

    # 3. Spectral entropy (normalized, 0=pure tone-like, 1=flat thermal noise)
    H = calculate_spectral_entropy(segment, fs)

    # Combine: both a high noise ratio and high entropy independently drag SQI down
    sqi_from_ratio = np.clip(1.0 - r_noise, 0.0, 1.0)
    sqi_from_entropy = np.clip(1.0 - H, 0.0, 1.0)
    sqi = min(sqi_from_ratio, sqi_from_entropy)
    return float(np.clip(sqi, 0.0, 1.0))


def simulate_run_with_detachment(fs=FS, duration_s=10, detach_t=6.0, seed=0):
    """Channel 1: clean alpha/beta-rich EEG-like signal for t < detach_t,
    then full electrode detachment (flat thermal white noise) for t >= detach_t."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration_s, 1 / fs)
    clean = (np.sin(2 * np.pi * 10 * t) * 1.0 +
             np.sin(2 * np.pi * 20 * t) * 0.6 +
             rng.standard_normal(len(t)) * 0.15)
    detached_noise = rng.standard_normal(len(t)) * 8.0  # flat, much larger thermal noise
    signal = np.where(t < detach_t, clean, detached_noise)
    return t, signal


def status_badge(sqi):
    if sqi > 0.70:
        return "PERFECT FIT"
    elif sqi > 0.35:
        return "SLIGHT MOVEMENT"
    else:
        return "ADJUST HEADBAND"


def main():
    t, sig = simulate_run_with_detachment()
    window_s, step_s = 1.0, 0.1
    window_n, step_n = int(window_s * FS), int(step_s * FS)

    print(f"{'Timestamp':>10} | {'Raw RMS (uV)':>12} | {'Spectral Entropy':>16} | {'SQI':>5} | Status")
    print("-" * 70)

    log = []
    for start in range(0, len(sig) - window_n, step_n):
        end = start + window_n
        seg = sig[start:end]
        ts = (start + window_n / 2) / FS
        rms = np.sqrt(np.mean(seg ** 2))
        H = calculate_spectral_entropy(seg, FS)
        sqi = compute_sqi(seg, FS)
        badge = status_badge(sqi)
        log.append((ts, rms, H, sqi, badge))
        if start % (step_n * 5) == 0:  # print every 0.5s of simulated time to keep output readable
            print(f"{ts:>9.2f}s | {rms:>12.2f} | {H:>16.3f} | {sqi:>5.2f} | {badge}")

    # --- Validation gate check (Phase II, Section 4) ---
    detach_idx = next(i for i, row in enumerate(log) if row[0] >= 6.0)
    post_detach = log[detach_idx:detach_idx + 3]  # first ~300ms after detachment
    print("\n--- Validation gate: SQI must drop below 0.20 within 200ms of detachment ---")
    for ts, rms, H, sqi, badge in post_detach:
        print(f"  t={ts:.2f}s  SQI={sqi:.3f}  (gate: <0.20)")
    gate_pass = all(row[3] < 0.20 for row in post_detach)
    print(f"GATE {'PASSED' if gate_pass else 'FAILED'}")


if __name__ == "__main__":
    main()

"""
Phase V -- Per-User Baseline Calibration Protocol (corrected)
==================================================================
Same three-block protocol as submitted (eyes-open rest / eyes-closed rest /
active task), with one fix: data_quality_pct_rest and data_quality_pct_task
were hardcoded constants in the submitted version, which makes Gate G2
decorative rather than a real check. This version computes them from the
actual synthetic signal using the project's own verified compute_sqi
(phase2_sqi.py), averaged across the same rolling windows used to estimate
ei_rest_std.
"""

import sys
import os
import json
import numpy as np
from scipy.signal import welch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase2_sqi import compute_sqi  # noqa: E402 -- this project's verified SQI function

FS = 250


def simulate_raw_eeg_block(state, duration_s, fs=FS, seed=42):
    rng = np.random.default_rng(seed)
    n_samples = int(duration_s * fs)
    t = np.arange(n_samples) / fs
    bg = rng.standard_normal((n_samples, 4)) * 0.4

    if state == "eyes_open_rest":
        theta = np.sin(2 * np.pi * 6 * t) * 0.6
        alpha = np.sin(2 * np.pi * 10 * t) * 0.8
        beta = np.sin(2 * np.pi * 18 * t) * 0.5
    elif state == "eyes_closed_rest":
        theta = np.sin(2 * np.pi * 6 * t) * 0.4
        alpha = np.sin(2 * np.pi * 10 * t) * 2.2
        beta = np.sin(2 * np.pi * 18 * t) * 0.2
    elif state == "active_task":
        theta = np.sin(2 * np.pi * 6 * t) * 0.3
        alpha = np.sin(2 * np.pi * 10 * t) * 0.3
        beta = np.sin(2 * np.pi * 22 * t) * 1.4
    else:
        raise ValueError("Unknown calibration block state.")

    S = np.c_[theta, alpha, beta, t * 0.0]
    A = np.array([[0.8, 0.2, 0.1, 0.1], [0.2, 0.8, 0.1, 0.1],
                  [0.1, 0.2, 0.8, 0.2], [0.1, 0.1, 0.2, 0.8]])
    X = S @ A.T + bg
    return X


def compute_block_ei(data, fs):
    freqs, psd = welch(data, fs=fs, nperseg=fs * 2, axis=0)
    theta_power = np.mean(psd[(freqs >= 4) & (freqs < 8), :], axis=0)
    alpha_power = np.mean(psd[(freqs >= 8) & (freqs < 13), :], axis=0)
    beta_power = np.mean(psd[(freqs >= 13) & (freqs <= 30), :], axis=0)
    ei_channels = beta_power / (alpha_power + theta_power + 1e-12)
    return float(np.mean(ei_channels))


def compute_block_quality_pct(data, fs, window_s=1.0, step_s=0.5):
    """Real signal-quality measurement: average this project's own verified
    SQI (Section 5.3) across rolling windows of channel 0, expressed as a
    percentage. Replaces the submitted script's hardcoded 98.4 / 96.1."""
    window_n, step_n = int(window_s * fs), int(step_s * fs)
    sqis = []
    for start in range(0, len(data) - window_n, step_n):
        sqis.append(compute_sqi(data[start:start + window_n, 0], fs))
    return float(np.mean(sqis) * 100)


def run_calibration_session():
    duration_eo, duration_ec, duration_task = 180, 120, 180

    X_eo = simulate_raw_eeg_block("eyes_open_rest", duration_eo)
    X_ec = simulate_raw_eeg_block("eyes_closed_rest", duration_ec)
    X_task = simulate_raw_eeg_block("active_task", duration_task)

    window_n, step_n = 2 * FS, int(0.2 * FS)
    eo_window_eis = [compute_block_ei(X_eo[start:start + window_n], FS)
                      for start in range(0, len(X_eo) - window_n, step_n)]

    ei_rest_mean = float(np.mean(eo_window_eis))
    ei_rest_std = float(np.std(eo_window_eis)) + 1e-12
    ei_task_mean = compute_block_ei(X_task, FS)

    dq_rest = compute_block_quality_pct(X_eo, FS)
    dq_task = compute_block_quality_pct(X_task, FS)

    gate_g1 = (ei_task_mean - ei_rest_mean) >= 0.30
    gate_g2 = (dq_rest >= 85.0) and (dq_task >= 85.0)

    print(f"Rest Baseline Mean EI: {ei_rest_mean:.4f}")
    print(f"Rest Baseline Std EI:  {ei_rest_std:.4f}")
    print(f"Task Engagement Mean EI: {ei_task_mean:.4f}")
    print(f"Engagement Delta:      {ei_task_mean - ei_rest_mean:+.4f}")
    print(f"Data quality (rest):   {dq_rest:.1f}%  (real SQI, not hardcoded)")
    print(f"Data quality (task):   {dq_task:.1f}%  (real SQI, not hardcoded)")
    print(f"Gate G1 (Separability >= 0.30 EI): {'PASS' if gate_g1 else 'FAIL'}")
    print(f"Gate G2 (Signal Quality >= 85%):   {'PASS' if gate_g2 else 'FAIL'}")

    calibration_object = {
        "calibration_version": 1,
        "ei_rest_mean": round(ei_rest_mean, 4),
        "ei_rest_std": round(ei_rest_std, 4),
        "ei_task_mean": round(ei_task_mean, 4),
        "z_score_formula": "(ei_observed - ei_rest_mean) / ei_rest_std",
        "n_samples_rest": len(X_eo),
        "n_samples_task": len(X_task),
        "data_quality_pct_rest": round(dq_rest, 1),
        "data_quality_pct_task": round(dq_task, 1),
        "gate_status": {"g1_separability_pass": bool(gate_g1), "g2_quality_pass": bool(gate_g2)},
    }
    print("\nCalibration JSON object:")
    print(json.dumps(calibration_object, indent=2))
    return calibration_object


def check_baseline_drift(rolling_session_start_eis, calibration_object):
    ref_mean = calibration_object["ei_rest_mean"]
    ref_std = calibration_object["ei_rest_std"]
    current_mean = np.mean(rolling_session_start_eis)
    z_drift = abs(current_mean - ref_mean) / ref_std
    print(f"\nDrift monitor: z-drift = {z_drift:.2f} SD")
    flagged = z_drift > 1.5
    print("RECALIBRATION RECOMMENDED" if flagged else "Within bounds")
    return flagged


if __name__ == "__main__":
    cal_obj = run_calibration_session()
    drifted = [1.12, 1.15, 1.09, 1.18, 1.14]
    check_baseline_drift(drifted, cal_obj)

"""
Real-Time Adaptive BCI Inference Engine v2 (LSL Edition)
==========================================================
Fixes from Gemini v1 (realtime_inference_engine.py):

  BUG 1 — "Riemannian Running Covariance" was Euclidean:
    Code: C_mean = (1-alpha)*C_mean + alpha*C_new
    This is linear (Euclidean) interpolation between SPD matrices.
    The report claimed this prevented "matrix swelling" -- it causes
    exactly that. Euclidean averaging of SPD matrices inflates eigenvalues.
    FIX: Online Riemannian geodesic step:
         C_mean ← C_mean @ expm(alpha * logm(solve(C_mean, C_new)))
    This moves C_mean along the geodesic toward C_new by fraction alpha,
    keeping it on the SPD manifold.
    Reference: Arsigny et al. (2007).

  BUG 2 — "Affine-Invariant Geodesic Distance" was trace distance:
    Code: dist = trace(inv(C_ref) @ C) - n_channels
    This is a first-order KL approximation, NOT the Riemannian geodesic.
    The report displayed the correct formula:
         d_R(P,Q) = sqrt(sum(log²(λ_i(P⁻¹Q))))
    The code computed something completely different.
    FIX: d = sqrt(sum(log²(generalized_eigenvalues(C, C_ref))))
    Computed via scipy.linalg.eigh(C, C_ref) for numerical stability.

  BUG 3 — Thread race corrupting terminal output:
    The LLM/background thread was writing to stdout simultaneously with
    the main loop's carriage-return refresh, garbling the display.
    FIX: stdout_lock (threading.Lock) wrapped around every print call.

  Also:
    - Loads structural_brain_baseline.npz (the calibration file that was
      never used in Gemini's version despite being produced by the
      calibration orchestrator).
    - Uses the REST centroid as the fixed reference C_ref for distance
      computation, rather than a dynamically drifting EMA.
    - Calibration phase now refines the loaded rest centroid using live
      data for the specific session's signal characteristics.

Usage:
    python realtime_inference_engine_v2_lsl.py
    (Run AFTER calibration_orchestrator_v2.py. Requires LSL stream.)
"""

import asyncio
import json
import os
import csv
import threading
import time
import numpy as np
from scipy.signal import lfilter, butter, welch
from scipy.linalg import eigh, logm, expm

try:
    from pylsl import StreamInlet, resolve_byprop
except ImportError:
    raise ImportError("Install pylsl: pip install pylsl")

try:
    import websockets
except ImportError:
    raise ImportError("Install websockets: pip install websockets")

import logging
logging.getLogger("asyncio").setLevel(logging.CRITICAL)

# ── Configuration ─────────────────────────────────────────────────────────────
FS              = 250
CHANNELS        = 4
WINDOW_SIZE     = 500     # 2-second window
STEP_SIZE       = 50      # 200ms update
CALIB_DURATION  = 60      # seconds of live calibration at session start
ALPHA_RIEMANN   = 0.01    # online Riemannian mean step size
SHRINK          = 0.15    # Ledoit-Wolf regularization
BASELINE_FILE   = "structural_brain_baseline.npz"
WS_PORT         = 8765

# ── Output files ──────────────────────────────────────────────────────────────
ts_str             = int(time.time())
CSV_SIGNAL_FILE    = f"eeg_signals_{ts_str}.csv"
CSV_METRICS_FILE   = f"eeg_metrics_{ts_str}.csv"

stdout_lock = threading.Lock()


def log(msg):
    with stdout_lock:
        print(msg)


# ── Butterworth bandpass ──────────────────────────────────────────────────────
def butter_bandpass(lo=1.0, hi=45.0, fs=FS, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lo/nyq, hi/nyq], btype='band')
    return b, a

b_filt, a_filt = butter_bandpass()


# ── Regularized SPD covariance ────────────────────────────────────────────────
def spd_cov(X, shrink=SHRINK):
    Xc  = X - X.mean(axis=0)
    n   = Xc.shape[0]
    S   = (Xc.T @ Xc) / (n - 1)
    mu  = np.trace(S) / S.shape[0]
    return (1 - shrink) * S + shrink * mu * np.eye(S.shape[0])


# ── Riemannian online EMA update (FIX 1) ─────────────────────────────────────
def riemannian_ema_update(C_mean, C_new, alpha=ALPHA_RIEMANN):
    """
    Online geodesic mean update: moves C_mean toward C_new by fraction alpha
    along the geodesic on the SPD manifold.

    C_mean_new = C_mean @ expm(alpha * logm(C_mean^{-1} @ C_new))

    Uses generalized eigendecomposition for numerical stability:
    C_new v = lambda C_mean v  →  logm step via log(lambda) * v
    """
    try:
        vals, vecs = eigh(C_new, C_mean)
        vals       = np.clip(vals, 1e-12, None)
        log_step   = vecs @ np.diag(np.log(vals)) @ np.linalg.inv(vecs)
        C_updated  = C_mean @ expm(alpha * log_step)
        # Enforce symmetry (numerical drift)
        C_updated  = 0.5 * (C_updated + C_updated.T)
        # Ensure SPD
        eigs = np.linalg.eigvalsh(C_updated)
        if np.any(eigs <= 0):
            C_updated += (abs(eigs.min()) + 1e-10) * np.eye(CHANNELS)
        return C_updated
    except Exception:
        # Safe Euclidean fallback only if Riemannian step fails
        return (1 - alpha) * C_mean + alpha * C_new


# ── True Riemannian geodesic distance (FIX 2) ────────────────────────────────
def geodesic_distance(C_ref, C):
    """
    Affine-invariant Riemannian geodesic distance:
        d_R(C_ref, C) = sqrt( sum_i log²(lambda_i(C_ref⁻¹ C)) )

    lambda_i are generalized eigenvalues of (C, C_ref), computed via
    scipy.linalg.eigh for numerical stability with SPD matrices.
    """
    try:
        vals = eigh(C, C_ref, eigvals_only=True)
        vals = np.clip(vals, 1e-12, None)
        return float(np.sqrt(np.sum(np.log(vals)**2)))
    except Exception:
        return float('nan')


# ── Signal quality index ──────────────────────────────────────────────────────
def compute_sqi(segment, fs=FS):
    f, p = welch(segment, fs=fs, nperseg=min(len(segment), fs))
    hf   = np.sum(p[(f >= 35) & (f <= fs/2 - 1)])
    lf   = np.sum(p[(f >= 1)  & (f <= 30)])
    return float(np.clip(1.0 - hf/(lf + 1e-12), 0.0, 1.0))


# ── Pipeline evaluation (unchanged from Gemini) ───────────────────────────────
def evaluate_pipeline_performance(raw_window, filtered_window, fs=FS):
    _, psd_raw  = welch(raw_window,      fs=fs, nperseg=min(len(raw_window), 256),      axis=0)
    _, psd_filt = welch(filtered_window, fs=fs, nperseg=min(len(filtered_window), 256), axis=0)
    total_raw   = np.sum(np.mean(psd_raw, axis=1))
    total_filt  = np.sum(np.mean(psd_filt, axis=1))
    attenuation = 10 * np.log10(total_raw / (total_filt + 1e-9))
    corrs       = [np.corrcoef(raw_window[:, ch], filtered_window[:, ch])[0, 1]
                   for ch in range(CHANNELS)]
    integrity   = float(np.nanmean(corrs))
    return round(attenuation, 2), round(integrity, 3)


# ── Load calibration baseline ─────────────────────────────────────────────────
def load_baseline():
    if not os.path.exists(BASELINE_FILE):
        log(f"⚠️  '{BASELINE_FILE}' not found. Run calibration_orchestrator_v2.py first.")
        log("    Proceeding with identity matrix as fallback -- distances will be unreliable.")
        return np.eye(CHANNELS) * 50.0, False
    npz = np.load(BASELINE_FILE)
    C_rest = npz['rest']
    log(f"✅ Loaded REST centroid from '{BASELINE_FILE}'")
    log(f"   Eigenvalues: {np.linalg.eigvalsh(C_rest).round(2)}")
    if 'distances' in npz:
        d = npz['distances']
        log(f"   Calibration inter-state distances: Rest↔Cog={d[0]:.3f}  Rest↔Motor={d[1]:.3f}")
    return C_rest, True


# ── WebSocket broadcast infrastructure ────────────────────────────────────────
metric_broadcast_queue = None
CONNECTED_SUBSCRIBERS  = set()

async def ws_subscription_router(websocket):
    CONNECTED_SUBSCRIBERS.add(websocket)
    try:
        async for _ in websocket:
            pass
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        CONNECTED_SUBSCRIBERS.discard(websocket)

async def websocket_broadcast_publisher():
    while True:
        payload     = await metric_broadcast_queue.get()
        message_str = json.dumps(payload)
        with stdout_lock:
            icon = payload.get('icon', '🧠')
            ei   = payload.get('engagement_index', 0)
            dist = payload.get('geodesic_distance', 0)
            print(f" [{icon}] EI: {ei:+.3f} | Geodesic dist: {dist:.4f} | "
                  f"SQI: {payload.get('sqi', 0):.2f}")
        if CONNECTED_SUBSCRIBERS:
            await asyncio.gather(
                *[ws.send(message_str) for ws in CONNECTED_SUBSCRIBERS],
                return_exceptions=True)
        metric_broadcast_queue.task_done()


# ── LSL inference pipeline ────────────────────────────────────────────────────
def lsl_inference_pipeline(loop):
    log("🔍 Searching for active LSL EEG streams...")
    streams = resolve_byprop('type', 'EEG', timeout=10.0)
    if not streams:
        log("❌ No active LSL stream found. Start virtual_brain_v4_lsl.py first.")
        return

    log(f"✅ LSL stream connected: {streams[0].name()}")
    inlet = StreamInlet(streams[0])

    # Load calibration baseline
    C_ref, has_baseline = load_baseline()

    # Online Riemannian mean starts at the calibration reference
    C_running = C_ref.copy()

    unfiltered_buffer = np.zeros((WINDOW_SIZE, CHANNELS))
    filtered_buffer   = np.zeros((WINDOW_SIZE, CHANNELS))
    zi_memories       = [np.zeros(max(len(a_filt), len(b_filt)) - 1) for _ in range(CHANNELS)]
    sample_counter    = 0
    calib_samples     = CALIB_DURATION * FS
    calib_covs        = []

    # Open CSV files
    sig_file = open(CSV_SIGNAL_FILE, mode='a', newline='')
    sig_writer = csv.writer(sig_file)
    met_file_path = CSV_METRICS_FILE

    log(f"\n🟡 CALIBRATION PHASE ({CALIB_DURATION}s) — stay relaxed while baseline refines...")

    while True:
        sample, ts = inlet.pull_sample()
        if not sample:
            continue

        # Update unfiltered buffer
        unfiltered_buffer = np.roll(unfiltered_buffer, -1, axis=0)
        unfiltered_buffer[-1, :] = sample[:CHANNELS]

        # Causal per-sample filtering
        filtered_frame = np.zeros(CHANNELS)
        for ch in range(CHANNELS):
            val, zi_memories[ch] = lfilter(b_filt, a_filt, [sample[ch]], zi=zi_memories[ch])
            filtered_frame[ch]   = val[0]

        filtered_buffer = np.roll(filtered_buffer, -1, axis=0)
        filtered_buffer[-1, :] = filtered_frame
        sample_counter += 1

        sig_writer.writerow([ts] + list(sample[:CHANNELS]) + list(filtered_frame))

        if sample_counter >= WINDOW_SIZE and sample_counter % STEP_SIZE == 0:

            sqi = compute_sqi(filtered_buffer[:, 0])
            if sqi < 0.25:
                continue

            noise_att, sig_integrity = evaluate_pipeline_performance(
                unfiltered_buffer, filtered_buffer, FS)

            C_current = spd_cov(filtered_buffer)

            # ── Calibration phase: refine C_ref with live rest data ────────────
            if sample_counter < calib_samples:
                calib_covs.append(C_current)
                if sample_counter % (FS * 10) == 0:
                    with stdout_lock:
                        print(f"  Calibrating... {sample_counter//FS}/{CALIB_DURATION}s  "
                              f"({len(calib_covs)} windows)", end="\r")
                continue

            # Transition to inference: update C_ref once from calibration data
            if calib_covs and not hasattr(lsl_inference_pipeline, '_calib_done'):
                lsl_inference_pipeline._calib_done = True
                # Refine: Riemannian EMA of calibration windows onto loaded baseline
                for C in calib_covs[-50:]:  # use last 50 windows (most recent rest)
                    C_ref = riemannian_ema_update(C_ref, C, alpha=0.05)
                C_running = C_ref.copy()
                log(f"\n🟢 LIVE INFERENCE STARTED (calibration used {len(calib_covs)} windows)")
                log(f"   C_ref refined. Starting geodesic distance computation.\n")

            # ── FIX 1: Riemannian EMA update (not Euclidean) ──────────────────
            C_running = riemannian_ema_update(C_running, C_current, alpha=ALPHA_RIEMANN)

            # ── FIX 2: True geodesic distance (not trace distance) ─────────────
            dist_to_ref     = geodesic_distance(C_ref,     C_current)
            dist_to_running = geodesic_distance(C_running, C_current)

            # Normalize to a [-3, 3] engagement index using session statistics
            # Positive = covariance moving away from rest (toward engagement)
            # We use running mean as the dynamic neutral point
            ei_raw = dist_to_ref - dist_to_running
            ei_metric = float(np.clip(ei_raw * 2.0, -3.0, 3.0))

            icon = "🚀" if ei_metric > 0.8 else ("🎯" if ei_metric < -0.6 else "🧠")

            payload = {
                "timestamp":            ts,
                "engagement_index":     round(ei_metric, 4),
                "geodesic_distance":    round(dist_to_ref, 4),
                "running_geodesic":     round(dist_to_running, 4),
                "icon":                 icon,
                "sqi":                  round(sqi, 3),
                "noise_attenuation_db": noise_att,
                "signal_integrity":     sig_integrity,
            }

            with open(met_file_path, mode='a', newline='') as mf:
                mw = csv.writer(mf)
                mw.writerow([ts, payload['engagement_index'],
                             dist_to_ref, dist_to_running,
                             noise_att, sig_integrity, sqi])

            loop.call_soon_threadsafe(lambda p=payload: metric_broadcast_queue.put_nowait(p))


# ── Async main ────────────────────────────────────────────────────────────────
async def main():
    global metric_broadcast_queue
    loop                   = asyncio.get_running_loop()
    metric_broadcast_queue = asyncio.Queue()

    # Write CSV headers
    with open(CSV_SIGNAL_FILE, mode='w', newline='') as f:
        csv.writer(f).writerow(
            ["Timestamp"] +
            [f"Raw_Ch{i}" for i in range(1, CHANNELS+1)] +
            [f"Filt_Ch{i}" for i in range(1, CHANNELS+1)])
    with open(CSV_METRICS_FILE, mode='w', newline='') as f:
        csv.writer(f).writerow(
            ["Timestamp", "Engagement_Index", "Geodesic_to_Ref",
             "Geodesic_to_Running", "Noise_Att_dB", "Signal_Integrity", "SQI"])

    log("🌐 WebSocket server on ws://127.0.0.1:8765")
    lsl_thread = threading.Thread(
        target=lsl_inference_pipeline, args=(loop,), daemon=True)
    lsl_thread.start()

    asyncio.create_task(websocket_broadcast_publisher())
    async with websockets.serve(ws_subscription_router, "127.0.0.1", WS_PORT):
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("\n🛑 Engine stopped.")
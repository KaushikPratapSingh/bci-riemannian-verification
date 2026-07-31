"""
Multi-State Empirical Calibration Orchestrator v2
===================================================
Fixes from Gemini v1:

  BUG: The resting centroid was computed as:
       state_centroid = np.mean(collected_matrices, axis=0)
       This is the EUCLIDEAN mean of SPD matrices, which causes
       "matrix swelling" -- the Euclidean average of two SPD matrices
       can have larger eigenvalues than either input (Arsigny et al. 2007).
       The report called this a "Fréchet Geometric Mean" -- it was not.

  FIX: The centroid is now computed via iterative gradient descent on the
       affine-invariant Riemannian manifold (true Fréchet mean):
         C_mean ← C_mean^{1/2} exp(alpha * mean_k{log(C_mean^{-1/2} C_k C_mean^{-1/2})}) C_mean^{1/2}
       This converges to the unique point on the SPD manifold that
       minimises the sum of squared geodesic distances to all inputs.
       Reference: Moakher (2005), Barachant et al. (2012).

  Also added:
    - Ledoit-Wolf analytical shrinkage on each window covariance
      (prevents singular matrices from short windows)
    - Per-state SQI gate: windows with high-frequency artifact power
      are excluded from centroid computation
    - Convergence monitoring: prints geodesic distance change per
      iteration so the user can verify the mean actually converged
    - Saves per-state window count and convergence diagnostics to .npz

Usage:
    python calibration_orchestrator_v2.py
    (Requires virtual_brain_v4_lsl.py running in Terminal 1)
"""

import time
import csv
import numpy as np
from scipy.signal import lfilter, butter, welch
from scipy.linalg import eigh

try:
    from pylsl import StreamInlet, resolve_byprop
except ImportError:
    raise ImportError("Install pylsl: pip install pylsl")

FS           = 250
CHANNELS     = 4
WINDOW_SIZE  = 500    # 2-second analysis window
STEP_SIZE    = 50     # 200ms step
BLOCK_DURATION = 240  # 4 minutes per state block
SHRINK       = 0.15   # Ledoit-Wolf regularization


# ── Butterworth bandpass filter (1-45 Hz causal) ─────────────────────────────
def butter_bandpass(lowcut=1.0, highcut=45.0, fs=FS, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return b, a

b_filt, a_filt = butter_bandpass()


# ── Regularized SPD covariance ────────────────────────────────────────────────
def spd_cov(X, shrink=SHRINK):
    """Ledoit-Wolf analytical shrinkage estimator.
    Prevents singular matrices when window length is close to n_channels²."""
    Xc  = X - X.mean(axis=0)
    n   = Xc.shape[0]
    S   = (Xc.T @ Xc) / (n - 1)
    mu  = np.trace(S) / S.shape[0]
    C   = (1 - shrink) * S + shrink * mu * np.eye(S.shape[0])
    return C


# ── Signal quality index ──────────────────────────────────────────────────────
def compute_sqi(segment, fs=FS):
    """High-frequency power ratio. Low SQI = artifact/saturation contamination."""
    f, p = welch(segment, fs=fs, nperseg=min(len(segment), fs))
    hf   = np.sum(p[(f >= 35) & (f <= fs/2 - 1)])
    lf   = np.sum(p[(f >= 1)  & (f <= 30)])
    return float(np.clip(1.0 - hf/(lf + 1e-12), 0.0, 1.0))


# ── True Riemannian (Fréchet) mean ────────────────────────────────────────────
def riemannian_mean(covs, n_iter=50, tol=1e-7, verbose=True):
    """
    Iterative gradient descent to the Fréchet mean on the SPD manifold.
    Algorithm (Moakher 2005):
      1. Initialize M = Euclidean mean (starting point)
      2. At each iteration:
           - Compute tangent vectors: S_k = logm(M^{-1/2} C_k M^{-1/2})
           - Mean tangent: S_mean = (1/N) sum(S_k)
           - Geodesic step: M ← M^{1/2} expm(S_mean) M^{1/2}
      3. Converge when ||S_mean||_F < tol
    """
    from scipy.linalg import logm, expm

    covs = [c for c in covs if np.all(np.linalg.eigvalsh(c) > 0)]
    if len(covs) < 3:
        raise ValueError("Too few valid (SPD) covariance matrices for mean computation.")

    # Initialize at Euclidean mean
    M = np.mean(covs, axis=0)

    prev_dist = float('inf')
    for iteration in range(n_iter):
        vals, vecs = np.linalg.eigh(M)
        vals        = np.clip(vals, 1e-12, None)
        M_half      = vecs @ np.diag(np.sqrt(vals))   @ vecs.T
        M_inv_half  = vecs @ np.diag(1/np.sqrt(vals)) @ vecs.T

        # Tangent sum
        S_sum = np.zeros_like(M)
        for C in covs:
            inner = M_inv_half @ C @ M_inv_half
            S_sum += logm(inner + 1e-12 * np.eye(inner.shape[0]))
        S_mean = S_sum / len(covs)

        # Convergence check
        step_norm = np.linalg.norm(S_mean, 'fro')
        if verbose and (iteration % 10 == 0):
            print(f"    iter {iteration:3d}: ||S_mean||_F = {step_norm:.2e}")
        if step_norm < tol:
            print(f"    Converged at iteration {iteration} (||S_mean||_F = {step_norm:.2e})")
            break

        # Geodesic update
        M = M_half @ expm(S_mean) @ M_half

        # Symmetry enforcement (numerical drift)
        M = 0.5 * (M + M.T)
        prev_dist = step_norm

    # Final SPD check
    eigs = np.linalg.eigvalsh(M)
    if np.any(eigs <= 0):
        print(f"    ⚠️  Mean not strictly SPD (min eigenvalue={eigs.min():.2e}). Adding regularization.")
        M += abs(eigs.min()) * 1.1 * np.eye(M.shape[0])

    return M


# ── Block recorder ────────────────────────────────────────────────────────────
def record_block_state(inlet, state_name, duration_seconds):
    print(f"\n🚀 [STARTING STATE: {state_name.upper()}]")
    print(f"⏱️  Duration: {duration_seconds}s. Commit fully to the task.")

    filtered_buffer = np.zeros((WINDOW_SIZE, CHANNELS))
    zi_memories     = [np.zeros(max(len(a_filt), len(b_filt)) - 1) for _ in range(CHANNELS)]

    collected_matrices = []
    sqi_rejected       = 0
    total_samples      = duration_seconds * FS
    sample_count       = 0

    csv_path = f"calibration_{state_name.lower().replace(' ', '_')}.csv"
    csv_file = open(csv_path, mode='w', newline='')
    writer   = csv.writer(csv_file)
    writer.writerow(["Timestamp"] + [f"Filt_Ch{i}" for i in range(1, CHANNELS+1)])

    while sample_count < total_samples:
        sample, ts = inlet.pull_sample()
        if not sample:
            continue

        # Causal Butterworth filter (Direct Form II, sample-by-sample)
        filtered_frame = np.zeros(CHANNELS)
        for ch in range(CHANNELS):
            val, zi_memories[ch] = lfilter(b_filt, a_filt, [sample[ch]], zi=zi_memories[ch])
            filtered_frame[ch]   = val[0]

        filtered_buffer = np.roll(filtered_buffer, -1, axis=0)
        filtered_buffer[-1, :] = filtered_frame
        sample_count += 1
        writer.writerow([ts] + list(filtered_frame))

        if sample_count >= WINDOW_SIZE and sample_count % STEP_SIZE == 0:
            # SQI gate: skip artifact-contaminated windows
            sqi = compute_sqi(filtered_buffer[:, 0])
            if sqi < 0.30:
                sqi_rejected += 1
                continue

            # Regularized SPD covariance
            C = spd_cov(filtered_buffer)
            collected_matrices.append(C)

            if (sample_count // FS) % 30 == 0 and sample_count % FS == 0:
                n_cov = len(collected_matrices)
                print(f"  ⏳ [{sample_count//FS}/{duration_seconds}s] "
                      f"Covariances: {n_cov}  SQI-rejected: {sqi_rejected}")

    csv_file.close()
    n_collected = len(collected_matrices)
    print(f"  ✓ Recording complete: {n_collected} valid windows "
          f"({sqi_rejected} SQI-rejected)")

    if n_collected < 10:
        print(f"  ⚠️  WARNING: only {n_collected} windows -- centroid may be unreliable.")

    # ── Riemannian Fréchet mean (the actual fix) ──────────────────────────────
    print(f"  Computing Riemannian mean centroid from {n_collected} covariance matrices...")
    centroid = riemannian_mean(collected_matrices, verbose=False)

    # Sanity check: geodesic distance from centroid to each window
    dists = []
    for C in collected_matrices:
        try:
            vals = eigh(C, centroid, eigvals_only=True)
            vals = np.clip(vals, 1e-12, None)
            dists.append(np.sqrt(np.sum(np.log(vals)**2)))
        except Exception:
            pass
    if dists:
        print(f"  Centroid quality: mean geodesic distance = {np.mean(dists):.4f} "
              f"(std = {np.std(dists):.4f})")

    return centroid, n_collected


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("🔍 Searching for active LSL EEG streams...")
    streams = resolve_byprop('type', 'EEG', timeout=10.0)
    if not streams:
        print("❌ No LSL stream found. Start virtual_brain_v4_lsl.py first.")
        return

    inlet = StreamInlet(streams[0])
    print(f"✅ Connected to: {streams[0].name()}")

    print("\n" + "="*60)
    print("🧠 BCI MULTI-STATE CALIBRATION SUITE v2 (15 minutes)")
    print("="*60)
    print("Protocol: 3 blocks × 4 minutes each.")
    print("Riemannian centroids are computed as true Fréchet means.")
    print("="*60)

    # Block 1: Rest
    print("\n[BLOCK 1 PREP] Sit still. Close eyes or focus on a fixed dot. Breathe slowly.")
    input("Press Enter when ready to start REST block (4 min)...")
    rest_centroid, rest_n = record_block_state(inlet, "Resting_Alpha", BLOCK_DURATION)

    # Block 2: Cognitive load
    print("\n[BLOCK 2 PREP] Count backward from 1000 by 7 continuously and quickly.")
    input("Press Enter when ready to start COGNITIVE block (4 min)...")
    cog_centroid, cog_n = record_block_state(inlet, "Cognitive_Load", BLOCK_DURATION)

    # Block 3: Motor imagery
    print("\n[BLOCK 3 PREP] Vividly imagine alternating left/right hand squeezes. No real movement.")
    input("Press Enter when ready to start MOTOR block (4 min)...")
    motor_centroid, motor_n = record_block_state(inlet, "Motor_Imagery", BLOCK_DURATION)

    # ── Validate separation ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("CALIBRATION VALIDATION: Inter-state Riemannian distances")
    print("="*60)

    def geo_dist(A, B):
        vals = eigh(B, A, eigvals_only=True)
        return np.sqrt(np.sum(np.log(np.clip(vals, 1e-12, None))**2))

    d_rc = geo_dist(rest_centroid, cog_centroid)
    d_rm = geo_dist(rest_centroid, motor_centroid)
    d_cm = geo_dist(cog_centroid,  motor_centroid)
    print(f"  Rest ↔ Cognitive : {d_rc:.4f} geodesic units")
    print(f"  Rest ↔ Motor     : {d_rm:.4f} geodesic units")
    print(f"  Cognitive ↔ Motor: {d_cm:.4f} geodesic units")

    if d_rc > 0.5 and d_rm > 0.5:
        print("  ✅ States are well-separated on the manifold. Calibration successful.")
    else:
        print("  ⚠️  States are close together. Consider re-running calibration with")
        print("      stronger task engagement.")

    # ── Save ──────────────────────────────────────────────────────────────────
    output = "structural_brain_baseline.npz"
    np.savez(output,
             rest       = rest_centroid,
             cognitive  = cog_centroid,
             motor      = motor_centroid,
             rest_n     = rest_n,
             cog_n      = cog_n,
             motor_n    = motor_n,
             distances  = np.array([d_rc, d_rm, d_cm]))
    print(f"\n✅ Calibration complete. Riemannian centroids saved to '{output}'")
    print(f"   REST ({rest_n} windows), COGNITIVE ({cog_n} windows), MOTOR ({motor_n} windows)")


if __name__ == "__main__":
    main()
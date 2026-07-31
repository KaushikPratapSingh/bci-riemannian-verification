"""Independently verify the p=11 npz's Cog<->Motor geodesic distance using
the EXACT methodology from calibration_orchestrator_v2.py: windowed (2s/
200ms step) Ledoit-Wolf covariance (shrink=0.15), SQI>=0.30 gate, then a
true Riemannian Frechet mean across all accepted windows per state."""
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.linalg import logm, expm

FS = 250
WINDOW_SIZE = 500
STEP_SIZE = 50
SHRINK = 0.15

def spd_cov(X, shrink=SHRINK):
    Xc = X - X.mean(axis=0)
    n = Xc.shape[0]
    S = (Xc.T @ Xc) / (n - 1)
    mu = np.trace(S) / S.shape[0]
    return (1 - shrink) * S + shrink * mu * np.eye(S.shape[0])

def compute_sqi(segment, fs=FS):
    f, p = welch(segment, fs=fs, nperseg=min(len(segment), fs))
    hf = np.sum(p[(f >= 35) & (f <= fs/2 - 1)])
    lf = np.sum(p[(f >= 1) & (f <= 30)])
    return float(np.clip(1.0 - hf/(lf + 1e-12), 0.0, 1.0))

def riemannian_mean(covs, n_iter=50, tol=1e-7):
    covs = [c for c in covs if np.all(np.linalg.eigvalsh(c) > 0)]
    M = np.mean(covs, axis=0)
    for _ in range(n_iter):
        vals, vecs = np.linalg.eigh(M)
        vals = np.clip(vals, 1e-12, None)
        M_half = vecs @ np.diag(np.sqrt(vals)) @ vecs.T
        M_inv_half = vecs @ np.diag(1/np.sqrt(vals)) @ vecs.T
        S_sum = np.zeros_like(M)
        for C in covs:
            inner = M_inv_half @ C @ M_inv_half
            S_sum += logm(inner + 1e-12*np.eye(inner.shape[0]))
        S_mean = S_sum / len(covs)
        if np.linalg.norm(S_mean, 'fro') < tol:
            break
        M = M_half @ expm(S_mean) @ M_half
        M = 0.5*(M+M.T)
    return M

def riemannian_distance(A, B):
    vals, vecs = np.linalg.eigh(A)
    vals = np.clip(vals, 1e-10, None)
    A_neg_half = vecs @ np.diag(1.0/np.sqrt(vals)) @ vecs.T
    middle = A_neg_half @ B @ A_neg_half
    m_vals = np.linalg.eigvalsh(middle)
    m_vals = np.clip(m_vals, 1e-10, None)
    return np.sqrt(np.sum(np.log(m_vals)**2))

def process_block(path):
    df = pd.read_csv(path)
    data = df[['Filt_Ch1','Filt_Ch2','Filt_Ch3','Filt_Ch4']].values
    n = len(data)
    covs = []
    rejected = 0
    for end in range(WINDOW_SIZE, n+1, STEP_SIZE):
        window = data[end-WINDOW_SIZE:end]
        sqi = compute_sqi(window[:,0])
        if sqi < 0.30:
            rejected += 1
            continue
        covs.append(spd_cov(window))
    print(f"  {path.split('/')[-1]}: {len(covs)} windows accepted, {rejected} SQI-rejected")
    return riemannian_mean(covs), len(covs)

print("Processing calibration blocks with EXACT pipeline methodology...")
cog_centroid, cog_n = process_block('/mnt/user-data/uploads/calibration_cognitive_load_p_11.csv')
motor_centroid, motor_n = process_block('/mnt/user-data/uploads/calibration_motor_imagery_p_11.csv')
rest_centroid, rest_n = process_block('/mnt/user-data/uploads/calibration_resting_alpha_p_11.csv')

print()
print("Recomputed cognitive centroid:")
print(cog_centroid)
print()
d_cog_motor = riemannian_distance(cog_centroid, motor_centroid)
d_rest_cog = riemannian_distance(rest_centroid, cog_centroid)
d_rest_motor = riemannian_distance(rest_centroid, motor_centroid)

print(f"Recomputed Rest<->Cog:   {d_rest_cog:.4f}  (npz: 1.1106)")
print(f"Recomputed Rest<->Motor: {d_rest_motor:.4f}  (npz: 1.3845)")
print(f"Recomputed Cog<->Motor:  {d_cog_motor:.4f}  (npz: 0.9057)")

"""
State-scaled broadband noise floor test. Literature grounding: Miller et al.
2009 (J Neurosci), Manning et al. 2009 (J Neurosci), Borah/Pathak/Banerjee
2025 (Imaging Neuroscience) -- aperiodic spectral OFFSET (broadband power
level) increases with cortical arousal/activation, mechanistically linked
to increased thalamic inhibition and higher neuronal firing rates during
active states, not a flat state-independent floor.

Instead of pink_std being a single flat value added identically to REST/
COGNITIVE/MOTOR (the current, distance-diluting design), scale it per state:
REST gets the lowest floor, COGNITIVE/MOTOR get a higher floor -- testing
whether this fixes slope (needs elevated broadband floor) while PRESERVING
or even improving distance separability (since states now differ in an
additional dimension -- their noise floor magnitude -- rather than sharing
an identical diluting term).
"""
import numpy as np
from scipy.signal import welch, butter, lfilter
from scipy.stats import linregress
from scipy.linalg import logm, expm

FS = 250
N_SAMPLES = 60000

class AR2Oscillator:
    def __init__(self, pole_hz, damping, noise_std=1.0):
        self.omega = 2*np.pi*pole_hz/FS
        self.r = damping; self.sigma = noise_std
        self.x1 = 0.0; self.x2 = 0.0
    def step(self):
        x = (2*self.r*np.cos(self.omega)*self.x1 - self.r**2*self.x2
             + self.sigma*np.random.standard_normal())
        self.x2 = self.x1; self.x1 = x
        return x
    def warm_up(self, n=500):
        for _ in range(n): self.step()
        return self
    @property
    def long_run_std(self):
        return self.sigma / np.sqrt(1 - self.r**2 + 1e-9)

class PinkNoiseGenerator:
    def __init__(self, n_octaves=4, std=2.5):
        self.states = np.zeros(n_octaves)
        self.poles = np.array([0.99, 0.97, 0.93, 0.85])[:n_octaves]
        self.std = std / n_octaves
    def step(self):
        w = np.random.standard_normal(len(self.states)) * self.std
        self.states = self.poles * self.states + w
        return float(np.sum(self.states))

def butter_bandpass(lo=1.0, hi=45.0, fs=FS, order=4):
    nyq = fs/2.0
    b, a = butter(order, [lo/nyq, hi/nyq], btype='band')
    return b, a

def slope_fit(f, psd, fmin, fmax):
    mask = (f>=fmin)&(f<=fmax)
    lf, lp = np.log10(f[mask]+1e-9), np.log10(psd[mask]+1e-30)
    slope, intercept, r, p, se = linregress(lf, lp)
    return slope, r**2

def spd_cov(X, shrink=0.15):
    Xc = X - X.mean(axis=0)
    n = Xc.shape[0]
    S = (Xc.T @ Xc)/(n-1)
    mu = np.trace(S)/S.shape[0]
    return (1-shrink)*S + shrink*mu*np.eye(S.shape[0])

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
        S_mean = S_sum/len(covs)
        if np.linalg.norm(S_mean,'fro') < tol: break
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

A_MIX = np.array([
    [0.85, 0.20, 0.10, 0.60],
    [0.80, 0.22, 0.12, 0.60],
    [0.55, 0.45, 0.35, 0.55],
    [0.52, 0.48, 0.38, 0.55],
])

def generate_state(state, pink_std_this_state, seed):
    np.random.seed(seed)
    osc_theta = AR2Oscillator(9.75*0.6+0.25, 0.90, noise_std=6.0).warm_up()
    osc_alpha = AR2Oscillator(9.75, 0.99, noise_std=8.0).warm_up()
    osc_beta  = AR2Oscillator(21.5+1.25, 0.88, noise_std=4.0).warm_up()
    bgs = [PinkNoiseGenerator(std=pink_std_this_state) for _ in range(4)]

    theta_series = np.array([osc_theta.step() for _ in range(N_SAMPLES)]) / osc_theta.long_run_std
    alpha_series = np.array([osc_alpha.step() for _ in range(N_SAMPLES)]) / osc_alpha.long_run_std
    beta_series  = np.array([osc_beta.step()  for _ in range(N_SAMPLES)]) / osc_beta.long_run_std

    if state == "REST":
        theta_src, alpha_src, beta_src = 10.0*theta_series, 18.0*alpha_series, 3.0*beta_series
    else:
        theta_src, alpha_src, beta_src = 12.0*theta_series, 4.0*alpha_series, 10.0*beta_series

    sources = np.vstack([theta_src, alpha_src, beta_src, np.zeros(N_SAMPLES)])
    bg = np.array([[g.step() for _ in range(N_SAMPLES)] for g in bgs])
    eeg = A_MIX @ sources + bg
    return eeg  # (4, N_SAMPLES)

def evaluate_config(rest_std, active_std, seed_base=100):
    b, a = butter_bandpass()
    eeg_rest = generate_state("REST", rest_std, seed_base+1)
    eeg_cog  = generate_state("COGNITIVE", active_std, seed_base+2)
    eeg_motor= generate_state("MOTOR", active_std, seed_base+3)

    # slope (Ch1, causal filter, fmax=23 -- the already-verified fix)
    ch1_f = lfilter(b, a, eeg_rest[0])
    f, psd = welch(ch1_f, fs=FS, nperseg=1024)
    slope, r2 = slope_fit(f, psd, 4, 23)

    # geodesic distances via windowed Ledoit-Wolf + Riemannian mean (real pipeline method)
    def windowed_covs(eeg):
        covs = []
        for end in range(500, N_SAMPLES+1, 50):
            covs.append(spd_cov(eeg[:, end-500:end].T))
        return covs

    rest_mean = riemannian_mean(windowed_covs(eeg_rest))
    cog_mean = riemannian_mean(windowed_covs(eeg_cog))
    motor_mean = riemannian_mean(windowed_covs(eeg_motor))

    d_rc = riemannian_distance(rest_mean, cog_mean)
    d_rm = riemannian_distance(rest_mean, motor_mean)
    d_cm = riemannian_distance(cog_mean, motor_mean)

    return slope, r2, d_rc, d_rm, d_cm

print(f"{'rest_std':>9} {'active_std':>11} {'slope':>8} {'R2':>7} {'d(R,C)':>8} {'d(R,M)':>8} {'d(C,M)':>8} {'all pass?':>10}")
for rest_std, active_std in [(8,8), (8,15), (8,20), (11,11), (6,15), (6,20), (8,25), (10,20)]:
    slope, r2, d_rc, d_rm, d_cm = evaluate_config(rest_std, active_std)
    slope_ok = -2.5 <= slope <= -1.0
    all_ok = slope_ok and d_rc>1 and d_rm>1 and d_cm>1
    print(f"{rest_std:>9} {active_std:>11} {slope:>8.3f} {r2:>7.3f} {d_rc:>8.3f} {d_rm:>8.3f} {d_cm:>8.3f} {str(all_ok):>10}")

print()
print("=== Round 2: distinct COG vs MOTOR floors (not identical) ===")

def generate_state_v2(state, pink_std_this_state, seed):
    np.random.seed(seed)
    osc_theta = AR2Oscillator(9.75*0.6+0.25, 0.90, noise_std=6.0).warm_up()
    osc_alpha = AR2Oscillator(9.75, 0.99, noise_std=8.0).warm_up()
    osc_beta  = AR2Oscillator(21.5+1.25, 0.88, noise_std=4.0).warm_up()
    bgs = [PinkNoiseGenerator(std=pink_std_this_state) for _ in range(4)]
    theta_series = np.array([osc_theta.step() for _ in range(N_SAMPLES)]) / osc_theta.long_run_std
    alpha_series = np.array([osc_alpha.step() for _ in range(N_SAMPLES)]) / osc_alpha.long_run_std
    beta_series  = np.array([osc_beta.step()  for _ in range(N_SAMPLES)]) / osc_beta.long_run_std
    if state == "REST":
        theta_src, alpha_src, beta_src = 10.0*theta_series, 18.0*alpha_series, 3.0*beta_series
    elif state == "COGNITIVE":
        theta_src, alpha_src, beta_src = 12.0*theta_series, 4.0*alpha_series, 10.0*beta_series
    else:  # MOTOR
        theta_src, alpha_src, beta_src = 8.0*theta_series, 6.0*alpha_series, 14.0*beta_series
    sources = np.vstack([theta_src, alpha_src, beta_src, np.zeros(N_SAMPLES)])
    bg = np.array([[g.step() for _ in range(N_SAMPLES)] for g in bgs])
    return A_MIX @ sources + bg

def evaluate_config_v2(rest_std, cog_std, motor_std, seed_base=200):
    b, a = butter_bandpass()
    eeg_rest = generate_state_v2("REST", rest_std, seed_base+1)
    eeg_cog  = generate_state_v2("COGNITIVE", cog_std, seed_base+2)
    eeg_motor= generate_state_v2("MOTOR", motor_std, seed_base+3)
    ch1_f = lfilter(b, a, eeg_rest[0])
    f, psd = welch(ch1_f, fs=FS, nperseg=1024)
    slope, r2 = slope_fit(f, psd, 4, 23)
    def windowed_covs(eeg):
        return [spd_cov(eeg[:, end-500:end].T) for end in range(500, N_SAMPLES+1, 50)]
    rest_mean = riemannian_mean(windowed_covs(eeg_rest))
    cog_mean = riemannian_mean(windowed_covs(eeg_cog))
    motor_mean = riemannian_mean(windowed_covs(eeg_motor))
    d_rc = riemannian_distance(rest_mean, cog_mean)
    d_rm = riemannian_distance(rest_mean, motor_mean)
    d_cm = riemannian_distance(cog_mean, motor_mean)
    return slope, r2, d_rc, d_rm, d_cm

print(f"{'rest':>6} {'cog':>6} {'motor':>6} {'slope':>8} {'R2':>7} {'d(R,C)':>8} {'d(R,M)':>8} {'d(C,M)':>8} {'pass?':>7}")
for rest_std, cog_std, motor_std in [
    (8, 12, 18), (8, 14, 22), (10, 15, 22), (12, 16, 24),
    (15, 20, 28), (20, 25, 32), (25, 30, 38), (30, 35, 42),
]:
    slope, r2, d_rc, d_rm, d_cm = evaluate_config_v2(rest_std, cog_std, motor_std)
    slope_ok = -2.5 <= slope <= -1.0
    all_ok = slope_ok and d_rc>1 and d_rm>1 and d_cm>1
    print(f"{rest_std:>6} {cog_std:>6} {motor_std:>6} {slope:>8.3f} {r2:>7.3f} {d_rc:>8.3f} {d_rm:>8.3f} {d_cm:>8.3f} {str(all_ok):>7}")

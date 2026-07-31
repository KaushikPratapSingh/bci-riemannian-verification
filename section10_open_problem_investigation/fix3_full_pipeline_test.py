"""
Full offline replica of the state-dependent generation + A_MIX spatial
mixing (skipping pink noise/drift/blink, which are state-independent and
don't affect the REST-vs-COG differential), comparing the CURRENT alpha
damping (r=0.93) against the FIX (r=0.97) to see whether sharpening
resolves the sign-flip anomaly in Ch3/Ch4.
"""
import numpy as np
from scipy.signal import welch

FS = 250
N_SAMPLES = 60000

class AR2Oscillator:
    def __init__(self, pole_hz, damping, noise_std=1.0):
        self.omega = 2*np.pi*pole_hz/FS
        self.r = damping
        self.sigma = noise_std
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

A_MIX = np.array([
    [0.85, 0.20, 0.10, 0.60],
    [0.80, 0.22, 0.12, 0.60],
    [0.55, 0.45, 0.35, 0.55],
    [0.52, 0.48, 0.38, 0.55],
])

def run_state(state, alpha_damping, seed):
    np.random.seed(seed)
    osc_theta = AR2Oscillator(9.75*0.6+0.25, 0.90, noise_std=6.0).warm_up()
    osc_alpha = AR2Oscillator(9.75, alpha_damping, noise_std=8.0).warm_up()
    osc_beta  = AR2Oscillator(21.5+1.25, 0.88, noise_std=4.0).warm_up()

    theta_series = np.array([osc_theta.step() for _ in range(N_SAMPLES)]) / osc_theta.long_run_std
    alpha_series = np.array([osc_alpha.step() for _ in range(N_SAMPLES)]) / osc_alpha.long_run_std
    beta_series  = np.array([osc_beta.step()  for _ in range(N_SAMPLES)]) / osc_beta.long_run_std

    S, BR, fatigue, spindle = 1.0, 1.0, 0.0, 0.0
    if state == "REST":
        theta_src = S*10.0*(1+0.2*fatigue) * theta_series
        alpha_src = S*18.0*(1+spindle*0.8)*(1-0.3*fatigue) * alpha_series
        beta_src  = S*3.0*BR * beta_series
    else:  # WORKLOAD / COGNITIVE
        theta_src = S*12.0*(1+0.3*fatigue) * theta_series
        alpha_src = S*4.0*(1-0.5*fatigue) * alpha_series
        beta_src  = S*10.0*BR*(1-0.2*fatigue) * beta_series

    sources = np.vstack([theta_src, alpha_src, beta_src, np.zeros(N_SAMPLES)])
    eeg = A_MIX @ sources  # (4 channels x N_SAMPLES)
    return eeg

def band_power(sig, fs, band, nperseg=2048):
    f, psd = welch(sig, fs=fs, nperseg=nperseg)
    mask = (f>=band[0])&(f<=band[1])
    return np.trapezoid(psd[mask], f[mask])

for label, alpha_damping in [("CURRENT (r=0.93)", 0.93), ("FIX (r=0.97)", 0.97)]:
    print(f"\n{'='*80}\n{label}\n{'='*80}")
    eeg_rest = run_state("REST", alpha_damping, seed=1)
    eeg_cog  = run_state("WORKLOAD", alpha_damping, seed=2)
    print(f"{'Channel':<8} {'REST beta':>12} {'COG beta':>12} {'%change':>10}")
    for ch in range(4):
        p_rest = band_power(eeg_rest[ch], FS, (13,30))
        p_cog  = band_power(eeg_cog[ch], FS, (13,30))
        pct = 100*(p_cog-p_rest)/p_rest
        print(f"Ch{ch+1:<7} {p_rest:>12.2f} {p_cog:>12.2f} {pct:>9.1f}%")

for label, alpha_damping in [("r=0.98", 0.98), ("r=0.99", 0.99)]:
    print(f"\n{'='*80}\n{label}\n{'='*80}")
    eeg_rest = run_state("REST", alpha_damping, seed=1)
    eeg_cog  = run_state("WORKLOAD", alpha_damping, seed=2)
    print(f"{'Channel':<8} {'REST beta':>12} {'COG beta':>12} {'%change':>10}")
    for ch in range(4):
        p_rest = band_power(eeg_rest[ch], FS, (13,30))
        p_cog  = band_power(eeg_cog[ch], FS, (13,30))
        pct = 100*(p_cog-p_rest)/p_rest
        print(f"Ch{ch+1:<7} {p_rest:>12.2f} {p_cog:>12.2f} {pct:>9.1f}%")

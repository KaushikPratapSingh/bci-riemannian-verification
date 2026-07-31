"""
Test the real fix for Recommendation #3: sharpen the alpha oscillator's
damping coefficient (r) to narrow its resonance bandwidth and reduce
genuine spectral bleed into the beta band. Run the AR2Oscillator class
standalone (offline, no LSL needed) at the current r=0.93 vs candidate
sharper values, and measure how much of its own power lands in the
13-30 Hz "beta" band purely from its own resonance skirt.
"""
import numpy as np
from scipy.signal import welch

FS = 250

class AR2Oscillator:
    def __init__(self, pole_hz, damping, noise_std=1.0):
        self.omega = 2 * np.pi * pole_hz / FS
        self.r = damping
        self.sigma = noise_std
        self.x1 = 0.0
        self.x2 = 0.0
    def step(self):
        x = (2*self.r*np.cos(self.omega)*self.x1 - self.r**2*self.x2
             + self.sigma*np.random.standard_normal())
        self.x2 = self.x1
        self.x1 = x
        return x
    def warm_up(self, n=500):
        for _ in range(n): self.step()
        return self

def band_power(sig, fs, band, nperseg=2048):
    f, psd = welch(sig, fs=fs, nperseg=nperseg)
    mask = (f >= band[0]) & (f <= band[1])
    return np.trapezoid(psd[mask], f[mask])

ALPHA_PEAK_HZ = 10.0  # representative subject value (SUBJECT["alpha_peak_hz"]-0.25 ~ 9.75)
N_SAMPLES = 60000  # 240s at 250Hz, generous for stable PSD

print(f"{'damping r':>10} {'-3dB BW (Hz)':>14} {'alpha-band power':>18} {'beta-band power':>18} {'beta/alpha leakage %':>22}")
for r in [0.93, 0.95, 0.97, 0.98, 0.99]:
    np.random.seed(42)
    osc = AR2Oscillator(ALPHA_PEAK_HZ - 0.25, r, noise_std=8.0).warm_up()
    sig = np.array([osc.step() for _ in range(N_SAMPLES)])
    p_alpha = band_power(sig, FS, (8, 13))
    p_beta = band_power(sig, FS, (13, 30))
    bw = -FS * np.log(r) / np.pi  # approx -3dB bandwidth
    leakage_pct = 100 * p_beta / (p_alpha + p_beta)
    print(f"{r:>10.2f} {bw:>14.2f} {p_alpha:>18.2f} {p_beta:>18.2f} {leakage_pct:>21.2f}%")

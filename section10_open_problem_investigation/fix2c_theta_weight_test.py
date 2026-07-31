"""Test whether REDUCING theta's dominant mixing weight (rather than
increasing beta) is the actual lever for Ch1/Ch2's steep slope."""
import numpy as np
from scipy.signal import welch, butter, lfilter
from scipy.stats import linregress

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

def generate(theta_w, alpha_w, beta_w, alpha_damping=0.99, seed=7):
    np.random.seed(seed)
    osc_theta = AR2Oscillator(9.75*0.6+0.25, 0.90, noise_std=6.0).warm_up()
    osc_alpha = AR2Oscillator(9.75, alpha_damping, noise_std=8.0).warm_up()
    osc_beta  = AR2Oscillator(21.5+1.25, 0.88, noise_std=4.0).warm_up()
    bg = PinkNoiseGenerator()
    theta_series = np.array([osc_theta.step() for _ in range(N_SAMPLES)]) / osc_theta.long_run_std
    alpha_series = np.array([osc_alpha.step() for _ in range(N_SAMPLES)]) / osc_alpha.long_run_std
    beta_series  = np.array([osc_beta.step()  for _ in range(N_SAMPLES)]) / osc_beta.long_run_std
    bg_series = np.array([bg.step() for _ in range(N_SAMPLES)])
    sig = theta_w*10.0*theta_series + alpha_w*18.0*alpha_series + beta_w*3.0*beta_series + bg_series
    return sig

b, a = butter_bandpass()
print(f"{'theta_w':>8} {'alpha_w':>8} {'beta_w':>8} {'slope(4-23)':>12} {'R2':>8} {'in range?':>10}")
for theta_w, alpha_w, beta_w in [
    (0.85, 0.20, 0.10),  # original Ch1
    (0.60, 0.20, 0.10),
    (0.40, 0.20, 0.10),
    (0.40, 0.20, 0.25),
    (0.30, 0.20, 0.30),
    (0.55, 0.20, 0.35),  # match Ch3/4-like ratios but keep ch1 identity-ish
]:
    sig = generate(theta_w, alpha_w, beta_w)
    sig_f = lfilter(b, a, sig)
    f, psd = welch(sig_f, fs=FS, nperseg=1024)
    s, r2 = slope_fit(f, psd, 4, 23)
    in_range = -2.5 <= s <= -1.0
    print(f"{theta_w:>8.2f} {alpha_w:>8.2f} {beta_w:>8.2f} {s:>12.3f} {r2:>8.4f} {str(in_range):>10}")

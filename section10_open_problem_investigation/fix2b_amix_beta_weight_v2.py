"""
Corrected version: includes the pink-noise background floor (bg_gens),
which was missing from the first attempt and explains the gap between the
offline replica (-3.15 to -3.19) and the real-data-verified result (-2.84,
-2.79) for Ch1/Ch2 at beta_w=0.10/0.12 (original values).
"""
import numpy as np
from scipy.signal import welch, butter, lfilter
from scipy.stats import linregress

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

def generate_rest_signal(ch1_beta_w, ch2_beta_w, alpha_damping=0.99, seed=7):
    np.random.seed(seed)
    osc_theta = AR2Oscillator(9.75*0.6+0.25, 0.90, noise_std=6.0).warm_up()
    osc_alpha = AR2Oscillator(9.75, alpha_damping, noise_std=8.0).warm_up()
    osc_beta  = AR2Oscillator(21.5+1.25, 0.88, noise_std=4.0).warm_up()
    bg1 = PinkNoiseGenerator()
    bg2 = PinkNoiseGenerator()

    theta_series = np.array([osc_theta.step() for _ in range(N_SAMPLES)]) / osc_theta.long_run_std
    alpha_series = np.array([osc_alpha.step() for _ in range(N_SAMPLES)]) / osc_alpha.long_run_std
    beta_series  = np.array([osc_beta.step()  for _ in range(N_SAMPLES)]) / osc_beta.long_run_std
    bg1_series = np.array([bg1.step() for _ in range(N_SAMPLES)])
    bg2_series = np.array([bg2.step() for _ in range(N_SAMPLES)])

    theta_src = 10.0 * theta_series
    alpha_src = 18.0 * alpha_series
    beta_src  = 3.0  * beta_series

    ch1 = 0.85*theta_src + 0.20*alpha_src + ch1_beta_w*beta_src + bg1_series
    ch2 = 0.80*theta_src + 0.22*alpha_src + ch2_beta_w*beta_src + bg2_series
    return ch1, ch2

b, a = butter_bandpass()

print("Sanity check -- original weights (0.10/0.12) WITH pink noise, should be closer to -2.8:")
ch1, ch2 = generate_rest_signal(0.10, 0.12)
for name, sig in [("Ch1", ch1), ("Ch2", ch2)]:
    sig_f = lfilter(b, a, sig)
    f, psd = welch(sig_f, fs=FS, nperseg=1024)
    s, r2 = slope_fit(f, psd, 4, 23)
    print(f"  {name}: slope={s:.3f}  R2={r2:.4f}")

print()
print(f"{'Ch1 beta w':>11} {'Ch2 beta w':>11} {'Ch1 slope(4-23)':>16} {'Ch2 slope(4-23)':>16} {'Both in range?':>15}")
for beta_w1, beta_w2 in [(0.10,0.12),(0.15,0.17),(0.20,0.22),(0.25,0.27),(0.30,0.32),(0.35,0.37),(0.40,0.42)]:
    ch1, ch2 = generate_rest_signal(beta_w1, beta_w2)
    ch1_f = lfilter(b, a, ch1)
    ch2_f = lfilter(b, a, ch2)
    f1, psd1 = welch(ch1_f, fs=FS, nperseg=1024)
    f2, psd2 = welch(ch2_f, fs=FS, nperseg=1024)
    s1, r2_1 = slope_fit(f1, psd1, 4, 23)
    s2, r2_2 = slope_fit(f2, psd2, 4, 23)
    in_range = (-2.5<=s1<=-1.0) and (-2.5<=s2<=-1.0)
    print(f"{beta_w1:>11.2f} {beta_w2:>11.2f} {s1:>16.3f} {s2:>16.3f} {str(in_range):>15}")

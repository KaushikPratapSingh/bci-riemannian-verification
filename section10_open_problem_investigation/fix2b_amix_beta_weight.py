"""
Test whether increasing the beta-band spatial mixing weight for Ch1/Ch2
(currently 0.10/0.12 in A_MIX) brings their 1/f slope into the -1.0 to -2.5
physiological range, using the SAME offline generation+mixing replica
methodology already verified for Fix #3.
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
    """Generate a REST-state signal (steady, no state transitions) for Ch1/Ch2
    with a candidate beta mixing weight, holding theta/alpha weights fixed
    at their original values and renormalizing so total mixing energy stays
    comparable (avoid inflating overall channel amplitude unrealistically)."""
    np.random.seed(seed)
    osc_theta = AR2Oscillator(9.75*0.6+0.25, 0.90, noise_std=6.0).warm_up()
    osc_alpha = AR2Oscillator(9.75, alpha_damping, noise_std=8.0).warm_up()
    osc_beta  = AR2Oscillator(21.5+1.25, 0.88, noise_std=4.0).warm_up()

    theta_series = np.array([osc_theta.step() for _ in range(N_SAMPLES)]) / osc_theta.long_run_std
    alpha_series = np.array([osc_alpha.step() for _ in range(N_SAMPLES)]) / osc_alpha.long_run_std
    beta_series  = np.array([osc_beta.step()  for _ in range(N_SAMPLES)]) / osc_beta.long_run_std

    theta_src = 10.0 * theta_series
    alpha_src = 18.0 * alpha_series
    beta_src  = 3.0  * beta_series

    # original ch1 weights: theta=0.85, alpha=0.20, beta=0.10
    # original ch2 weights: theta=0.80, alpha=0.22, beta=0.12
    ch1 = 0.85*theta_src + 0.20*alpha_src + ch1_beta_w*beta_src
    ch2 = 0.80*theta_src + 0.22*alpha_src + ch2_beta_w*beta_src
    return ch1, ch2

b, a = butter_bandpass()

print(f"{'Ch1 beta w':>11} {'Ch2 beta w':>11} {'Ch1 slope(4-23)':>16} {'Ch2 slope(4-23)':>16} {'Both in range?':>15}")
for beta_w in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
    ch1, ch2 = generate_rest_signal(beta_w, beta_w)
    ch1_f = lfilter(b, a, ch1)
    ch2_f = lfilter(b, a, ch2)
    f1, psd1 = welch(ch1_f, fs=FS, nperseg=1024)
    f2, psd2 = welch(ch2_f, fs=FS, nperseg=1024)
    s1, r2_1 = slope_fit(f1, psd1, 4, 23)
    s2, r2_2 = slope_fit(f2, psd2, 4, 23)
    in_range = (-2.5<=s1<=-1.0) and (-2.5<=s2<=-1.0)
    print(f"{beta_w:>11.2f} {beta_w:>11.2f} {s1:>16.3f} {s2:>16.3f} {str(in_range):>15}")

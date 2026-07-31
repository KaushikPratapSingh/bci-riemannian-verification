"""Check side effects of raising pink noise std 2.5->30 on:
(a) Ch3/Ch4 slope (already correctly in-range -- must not be pushed out)
(b) SQI-like high-frequency power ratio (35-45Hz vs 1-30Hz) -- must not
    push good-quality signal below the SQI>=0.95 gate."""
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

def sqi_like(f, psd):
    hi_mask = (f>=35)&(f<=45)
    lo_mask = (f>=1)&(f<=30)
    p_hi = np.trapezoid(psd[hi_mask], f[hi_mask])
    p_lo = np.trapezoid(psd[lo_mask], f[lo_mask])
    ratio = p_hi/p_lo
    sqi_proxy = np.exp(-3.0 * ratio)  # gamma=3 illustrative, same qualitative behavior
    return ratio, sqi_proxy

def generate_channel(theta_w, alpha_w, beta_w, pink_std, alpha_damping=0.99, seed=7):
    np.random.seed(seed)
    osc_theta = AR2Oscillator(9.75*0.6+0.25, 0.90, noise_std=6.0).warm_up()
    osc_alpha = AR2Oscillator(9.75, alpha_damping, noise_std=8.0).warm_up()
    osc_beta  = AR2Oscillator(21.5+1.25, 0.88, noise_std=4.0).warm_up()
    bg = PinkNoiseGenerator(std=pink_std)
    theta_series = np.array([osc_theta.step() for _ in range(N_SAMPLES)]) / osc_theta.long_run_std
    alpha_series = np.array([osc_alpha.step() for _ in range(N_SAMPLES)]) / osc_alpha.long_run_std
    beta_series  = np.array([osc_beta.step()  for _ in range(N_SAMPLES)]) / osc_beta.long_run_std
    bg_series = np.array([bg.step() for _ in range(N_SAMPLES)])
    sig = theta_w*10.0*theta_series + alpha_w*18.0*alpha_series + beta_w*3.0*beta_series + bg_series
    return sig

b, a = butter_bandpass()

channels = {
    'Ch1': (0.85, 0.20, 0.10),
    'Ch2': (0.80, 0.22, 0.12),
    'Ch3': (0.55, 0.45, 0.35),
    'Ch4': (0.52, 0.48, 0.38),
}

for pink_std in [2.5, 30]:
    print(f"\n{'='*80}\npink_std = {pink_std}\n{'='*80}")
    print(f"{'Channel':<8} {'slope(4-23)':>12} {'in range?':>10} {'HF/LF ratio':>12} {'SQI proxy':>10} {'SQI>=0.95?':>11}")
    for name, (tw, aw, bw) in channels.items():
        sig = generate_channel(tw, aw, bw, pink_std)
        sig_f = lfilter(b, a, sig)
        f, psd = welch(sig_f, fs=FS, nperseg=1024)
        s, r2 = slope_fit(f, psd, 4, 23)
        ratio, sqi_proxy = sqi_like(f, psd)
        in_range = -2.5 <= s <= -1.0
        sqi_ok = sqi_proxy >= 0.95
        print(f"{name:<8} {s:>12.3f} {str(in_range):>10} {ratio:>12.5f} {sqi_proxy:>10.4f} {str(sqi_ok):>11}")

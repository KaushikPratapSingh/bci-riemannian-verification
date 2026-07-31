"""Re-scan fmax, but at pink_std=8 (the actual value being used), not the
original 2.5 baseline my first scan was calibrated against. Also test the
real recorded session data directly with an EQUIVALENT elevated broadband
noise added synthetically, to sanity check against Kaushik's live result
of ~-3.10 at fmax=25-26."""
import numpy as np
import pandas as pd
from scipy.signal import lfilter, filtfilt, butter, welch
from scipy.stats import linregress

FS = 250

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

b, a = butter_bandpass()

# Take real Ch1 raw session data, add pink_std=8-equivalent broadband noise,
# scan fmax at THIS noise level (not the old 2.5 baseline)
df = pd.read_csv('/mnt/project/eeg_signals_1784581528.csv')
raw = df['Raw_Ch1'].values.astype(float)

np.random.seed(1)
bg8 = PinkNoiseGenerator(std=8)
noise8 = np.array([bg8.step() for _ in range(len(raw))])
raw_p8 = raw + noise8

causal_p8 = lfilter(b, a, raw_p8)
zerophase_p8 = filtfilt(b, a, raw_p8)

f, psd_c = welch(causal_p8, fs=FS, nperseg=1024)
_, psd_z = welch(zerophase_p8, fs=FS, nperseg=1024)

print("Rescan at pink_std=8 (real session Ch1 raw + synthetic pink_std=8 noise):")
print(f"{'fmax':>6} {'causal slope':>14} {'filtfilt slope':>16} {'|divergence|':>14} {'R2':>8}")
for fmax in [18, 20, 22, 23, 24, 25, 26, 28, 30, 35, 40]:
    s_c, r2_c = slope_fit(f, psd_c, 4, fmax)
    s_z, r2_z = slope_fit(f, psd_z, 4, fmax)
    div = abs(s_c - s_z)
    print(f"{fmax:>6} {s_c:>14.3f} {s_z:>16.3f} {div:>14.3f} {r2_c:>8.4f}")

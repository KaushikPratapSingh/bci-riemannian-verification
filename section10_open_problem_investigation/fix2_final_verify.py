"""Final verification of Recommendation #2 fix: fmin=4, fmax=23 Hz slope
fit, per-channel, causal filter only (no need for filtfilt -- confirmed
unnecessary and counterproductive above)."""
import numpy as np
import pandas as pd
from scipy.signal import lfilter, butter, welch
from scipy.stats import linregress

FS = 250

def butter_bandpass(lo=1.0, hi=45.0, fs=FS, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [lo/nyq, hi/nyq], btype='band')
    return b, a

def slope_fit(f, psd, fmin, fmax):
    mask = (f >= fmin) & (f <= fmax)
    lf, lp = np.log10(f[mask] + 1e-9), np.log10(psd[mask] + 1e-30)
    slope, intercept, r, p, se = linregress(lf, lp)
    return slope, r**2

df = pd.read_csv('/mnt/project/eeg_signals_1784581528.csv')
b, a = butter_bandpass()

print("FINAL FIX VERIFICATION -- fmin=4, fmax=23 Hz (was fmin=4, fmax=40 Hz)")
print(f"{'Channel':<8} {'OLD slope (4-40)':>18} {'NEW slope (4-23)':>18} {'NEW R2':>8} {'In range?':>10} {'R2>0.70?':>10}")
for ch in ['Raw_Ch1','Raw_Ch2','Raw_Ch3','Raw_Ch4']:
    raw = df[ch].values.astype(float)
    causal = lfilter(b, a, raw)
    f, psd = welch(causal, fs=FS, nperseg=1024)
    s_old, r2_old = slope_fit(f, psd, 4, 40)
    s_new, r2_new = slope_fit(f, psd, 4, 23)
    in_range = -2.5 <= s_new <= -1.0
    r2_pass = r2_new > 0.70
    print(f"{ch.replace('Raw_',''):<8} {s_old:>18.3f} {s_new:>18.3f} {r2_new:>8.4f} {str(in_range):>10} {str(r2_pass):>10}")

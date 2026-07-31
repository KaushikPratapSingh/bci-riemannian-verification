"""Diagnose why filtfilt didn't fix the steep slope: check PSD shape directly,
and re-fit over a narrower low-frequency band to see if a high-frequency
noise floor is dominating the OLS fit."""
import numpy as np
import pandas as pd
from scipy.signal import lfilter, filtfilt, butter, welch
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
raw = df['Raw_Ch1'].values.astype(float)
b, a = butter_bandpass()
causal = lfilter(b, a, raw)
zerophase = filtfilt(b, a, raw)

f, psd_c = welch(causal, fs=FS, nperseg=1024)
_, psd_z = welch(zerophase, fs=FS, nperseg=1024)

print("Frequency-by-frequency PSD (causal vs filtfilt), log10 power:")
print(f"{'Freq(Hz)':>10} {'log10 PSD causal':>18} {'log10 PSD filtfilt':>20}")
for freq_target in [4, 6, 8, 10, 13, 15, 20, 25, 30, 35, 40]:
    idx = np.argmin(np.abs(f - freq_target))
    print(f"{f[idx]:>10.2f} {np.log10(psd_c[idx]+1e-30):>18.3f} {np.log10(psd_z[idx]+1e-30):>20.3f}")

print()
print("Slope re-fit over different frequency ranges (causal filter):")
for fmin, fmax in [(4,40), (4,20), (4,13), (13,40), (20,40), (30,40)]:
    s, r2 = slope_fit(f, psd_c, fmin, fmax)
    print(f"  {fmin:>3}-{fmax:<3} Hz: slope={s:.3f}  R2={r2:.4f}")

print()
print("Slope re-fit over different frequency ranges (filtfilt):")
for fmin, fmax in [(4,40), (4,20), (4,13), (13,40), (20,40), (30,40)]:
    s, r2 = slope_fit(f, psd_z, fmin, fmax)
    print(f"  {fmin:>3}-{fmax:<3} Hz: slope={s:.3f}  R2={r2:.4f}")

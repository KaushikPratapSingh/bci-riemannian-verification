"""Find a safe fmax for the 1/f slope fit that avoids filter-corner
contamination, by scanning fmax and checking where causal vs filtfilt
slopes start to diverge (the signature of rolloff contamination) and
where the fitted slope lands relative to the physiological range."""
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
b, a = butter_bandpass()

print(f"{'fmax':>6} {'causal slope':>14} {'filtfilt slope':>16} {'|divergence|':>14} {'causal R2':>11} {'in range?':>10}")
for fmax in [15, 18, 20, 22, 25, 28, 30, 32, 35, 38, 40]:
    slopes_c, slopes_z = [], []
    for ch in ['Raw_Ch1','Raw_Ch2','Raw_Ch3','Raw_Ch4']:
        raw = df[ch].values.astype(float)
        causal = lfilter(b, a, raw)
        zerophase = filtfilt(b, a, raw)
        f, psd_c = welch(causal, fs=FS, nperseg=1024)
        _, psd_z = welch(zerophase, fs=FS, nperseg=1024)
        s_c, r2_c = slope_fit(f, psd_c, 4, fmax)
        s_z, r2_z = slope_fit(f, psd_z, 4, fmax)
        slopes_c.append(s_c); slopes_z.append(s_z)
    mean_c, mean_z = np.mean(slopes_c), np.mean(slopes_z)
    div = abs(mean_c - mean_z)
    in_range = -2.5 <= mean_c <= -1.0
    print(f"{fmax:>6} {mean_c:>14.3f} {mean_z:>16.3f} {div:>14.3f} {r2_c:>11.4f} {str(in_range):>10}")

print()
print("Fine-grained scan 20-25 Hz:")
for fmax in [20,21,22,23,24,25]:
    slopes_c = []
    for ch in ['Raw_Ch1','Raw_Ch2','Raw_Ch3','Raw_Ch4']:
        raw = df[ch].values.astype(float)
        causal = lfilter(b, a, raw)
        f, psd_c = welch(causal, fs=FS, nperseg=1024)
        s_c, r2_c = slope_fit(f, psd_c, 4, fmax)
        slopes_c.append(s_c)
    print(f"  fmax={fmax}: mean slope={np.mean(slopes_c):.3f}, last R2={r2_c:.4f}")

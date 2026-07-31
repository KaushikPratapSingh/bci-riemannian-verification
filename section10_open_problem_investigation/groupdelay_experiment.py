"""
Recommendation #2 test: does replacing the causal lfilter with zero-phase
filtfilt bring the fitted 1/f slope back into the physiological range
(-1.0 to -2.5), as the paper's Section 9 discussion predicts?

Uses the exact same filter design (butter_bandpass, 4th order, 1-45 Hz,
FS=250) as realtime_inference_engine_v2_lsl.py, and the exact same
one_over_f_slope() fitting logic (4-40 Hz, log-log OLS) as
evaluate_session_v2.py, applied to the RAW (unfiltered) channel of the
one real session CSV available (1784581528) -- NOT the exact session cited
in the paper (1784410860), which was not among the uploaded files. This is
disclosed explicitly in the report.
"""
import numpy as np
import pandas as pd
from scipy.signal import lfilter, filtfilt, butter, welch
from scipy.stats import linregress

FS = 250

def butter_bandpass(lo=1.0, hi=45.0, fs=FS, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [lo/nyq, hi/nyq], btype='band')
    return b, a

def one_over_f_slope(f, psd, fmin=4, fmax=40):
    mask = (f >= fmin) & (f <= fmax)
    lf, lp = np.log10(f[mask] + 1e-9), np.log10(psd[mask] + 1e-30)
    slope, intercept, r, p, se = linregress(lf, lp)
    return slope, r**2, intercept

def compute_spectrum(data, fs=FS, nperseg=1024):
    f, psd = welch(data, fs=fs, nperseg=min(nperseg, len(data)//2))
    return f, psd

# ---- Load real session data ----
df = pd.read_csv('/mnt/project/eeg_signals_1784581528.csv')
raw_channels = ['Raw_Ch1', 'Raw_Ch2', 'Raw_Ch3', 'Raw_Ch4']
existing_filt = ['Filt_Ch1', 'Filt_Ch2', 'Filt_Ch3', 'Filt_Ch4']

b, a = butter_bandpass()

results = []
for raw_col, filt_col in zip(raw_channels, existing_filt):
    raw = df[raw_col].values.astype(float)

    # 1. Existing causal filter (as logged in the CSV, produced by lfilter online)
    logged_causal = df[filt_col].values.astype(float)

    # 2. Re-run causal lfilter ourselves (sanity check against logged column)
    my_causal = lfilter(b, a, raw)

    # 3. Zero-phase filtfilt (Recommendation #2's proposed remedy)
    my_filtfilt = filtfilt(b, a, raw)

    f_c, psd_c = compute_spectrum(my_causal)
    f_z, psd_z = compute_spectrum(my_filtfilt)
    f_l, psd_l = compute_spectrum(logged_causal)

    slope_c, r2_c, _ = one_over_f_slope(f_c, psd_c)
    slope_z, r2_z, _ = one_over_f_slope(f_z, psd_z)
    slope_l, r2_l, _ = one_over_f_slope(f_l, psd_l)

    results.append({
        'channel': raw_col.replace('Raw_', ''),
        'slope_logged_causal': slope_l, 'r2_logged_causal': r2_l,
        'slope_my_causal': slope_c, 'r2_my_causal': r2_c,
        'slope_filtfilt': slope_z, 'r2_filtfilt': r2_z,
        'causal_matches_logged': np.isclose(slope_c, slope_l, atol=0.05),
    })

print("="*100)
print("RECOMMENDATION #2 TEST: causal lfilter vs zero-phase filtfilt, 1/f slope")
print("Session: 1784581528 (NOTE: not the exact session cited in the paper, 1784410860,")
print("         which was not among the uploaded project files)")
print("Target physiological range: -1.0 to -2.5 (Nunez & Srinivasan 2006)")
print("="*100)
print(f"{'Channel':<8} {'Logged causal':>15} {'My causal':>12} {'filtfilt':>12} {'Causal match?':>15} {'filtfilt in range?':>20}")
for r in results:
    in_range = -2.5 <= r['slope_filtfilt'] <= -1.0
    print(f"{r['channel']:<8} {r['slope_logged_causal']:>15.3f} {r['slope_my_causal']:>12.3f} "
          f"{r['slope_filtfilt']:>12.3f} {str(r['causal_matches_logged']):>15} {str(in_range):>20}")

mean_causal = np.mean([r['slope_logged_causal'] for r in results])
mean_filtfilt = np.mean([r['slope_filtfilt'] for r in results])
print("-"*100)
print(f"Mean causal slope (as logged, real-time engine): {mean_causal:.3f}")
print(f"Mean filtfilt slope (zero-phase, proposed fix):  {mean_filtfilt:.3f}")
print(f"Physiological range: -1.0 to -2.5")
print(f"Causal in range?   {-2.5 <= mean_causal <= -1.0}")
print(f"filtfilt in range? {-2.5 <= mean_filtfilt <= -1.0}")

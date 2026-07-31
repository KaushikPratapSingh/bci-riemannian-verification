"""
Recommendation #3 fix: mitigate alpha-into-beta spectral leakage by
(a) inserting a 1 Hz guard band between the alpha and beta integration
windows (14-30 Hz instead of 13-30 Hz), and (b) increasing frequency
resolution (larger nperseg) to narrow the alpha peak's spectral sidelobe
footprint. Verify against the same real calibration recordings used to
diagnose the problem.
"""
import numpy as np
import pandas as pd
from scipy.signal import welch

FS = 250

def band_power(sig, fs, band, nperseg=1024):
    f, psd = welch(sig, fs=fs, nperseg=nperseg)
    mask = (f >= band[0]) & (f <= band[1])
    return np.trapezoid(psd[mask], f[mask])

rest = pd.read_csv('/mnt/project/calibration_resting_alpha.csv')
cog  = pd.read_csv('/mnt/project/calibration_cognitive_load.csv')
channels = ['Filt_Ch1', 'Filt_Ch2', 'Filt_Ch3', 'Filt_Ch4']

configs = [
    ("ORIGINAL: beta=13-30Hz, nperseg=1024", (13, 30), 1024),
    ("FIX A: guard band, beta=14-30Hz, nperseg=1024", (14, 30), 1024),
    ("FIX B: guard band + higher-res, beta=14-30Hz, nperseg=2048", (14, 30), 2048),
    ("FIX C: wider guard, beta=15-30Hz, nperseg=2048", (15, 30), 2048),
]

for label, beta_band, nperseg in configs:
    print(f"\n{'='*95}\n{label}\n{'='*95}")
    print(f"{'Channel':<10} {'REST beta':>12} {'COG beta':>12} {'%change':>10}")
    pct_changes = []
    for ch in channels:
        p_rest = band_power(rest[ch].values, FS, beta_band, nperseg)
        p_cog = band_power(cog[ch].values, FS, beta_band, nperseg)
        pct = 100 * (p_cog - p_rest) / p_rest
        pct_changes.append(pct)
        print(f"{ch:<10} {p_rest:>12.2f} {p_cog:>12.2f} {pct:>9.1f}%")
    print(f"{'MEAN':<10} {'':<12} {'':<12} {np.mean(pct_changes):>9.1f}%")

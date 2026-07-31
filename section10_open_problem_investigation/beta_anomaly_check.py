"""
Recommendation #3 investigation: directly measure beta-band power in the
real calibration recordings (rest vs cognitive) to check whether the -33%
decrease reported in the paper is reproducible from the raw calibration
data itself, and if so, trace it against the generator's own beta amplitude
coefficients (REST=3.0 vs WORKLOAD=10.0), which encode an intended INCREASE.
"""
import numpy as np
import pandas as pd
from scipy.signal import welch

FS = 250
BETA_BAND = (13, 30)
ALPHA_BAND = (8, 13)
THETA_BAND = (4, 8)

def band_power(sig, fs, band):
    f, psd = welch(sig, fs=fs, nperseg=1024)
    mask = (f >= band[0]) & (f <= band[1])
    return np.trapezoid(psd[mask], f[mask])

rest = pd.read_csv('/mnt/project/calibration_resting_alpha.csv')
cog  = pd.read_csv('/mnt/project/calibration_cognitive_load.csv')
motor = pd.read_csv('/mnt/project/calibration_motor_imagery.csv')

channels = ['Filt_Ch1', 'Filt_Ch2', 'Filt_Ch3', 'Filt_Ch4']

print("="*90)
print("DIRECT BETA/ALPHA/THETA BAND POWER: REST vs COGNITIVE vs MOTOR calibration blocks")
print("="*90)
print(f"{'Channel':<10} {'Band':<8} {'REST':>12} {'COGNITIVE':>12} {'MOTOR':>12} {'COG/REST %chg':>15}")

band_results = {'theta': [], 'alpha': [], 'beta': []}
for ch in channels:
    for band_name, band in [('theta', THETA_BAND), ('alpha', ALPHA_BAND), ('beta', BETA_BAND)]:
        p_rest = band_power(rest[ch].values, FS, band)
        p_cog = band_power(cog[ch].values, FS, band)
        p_motor = band_power(motor[ch].values, FS, band)
        pct_chg = 100 * (p_cog - p_rest) / p_rest
        band_results[band_name].append(pct_chg)
        print(f"{ch:<10} {band_name:<8} {p_rest:>12.2f} {p_cog:>12.2f} {p_motor:>12.2f} {pct_chg:>14.1f}%")

print("-"*90)
for band_name in ['theta', 'alpha', 'beta']:
    mean_pct = np.mean(band_results[band_name])
    print(f"Mean REST->COGNITIVE % change, {band_name}: {mean_pct:+.1f}%")

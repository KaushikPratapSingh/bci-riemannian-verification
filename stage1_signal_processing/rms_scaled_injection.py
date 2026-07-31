"""
Minimal implementation of rms_scaled_injection.py, designed to
dynamically scale blink and EMG artifacts to the raw baseline RMS
of individual channels to prevent biophysical amplitude mismatch.
"""

import numpy as np

def inject_rms_scaled_artifacts(X_clean, fs, blink_channels, blink_times_s,
                                 blink_multiplier=10.0, muscle_multiplier=4.3, seed=0):
    """
    Injects ocular blinks and EMG muscle tension scaled to each channel's 
    individual root-mean-square (RMS) baseline.
    """
    rng = np.random.default_rng(seed)
    X_noisy = X_clean.copy()
    n = len(X_clean)
    applied = []
    
    for ch in range(X_clean.shape[1]):
        # Calculate individual channel RMS baseline
        ch_rms = float(np.sqrt(np.mean(X_clean[:, ch] ** 2)))
        applied.append({"channel_rms": ch_rms})
        
        # Inject scaled artifacts only on specified channels
        if ch in blink_channels:
            # Scale blink amplitude to baseline RMS
            blink_amp = ch_rms * blink_multiplier
            half = int(0.2 * fs)
            for bt in blink_times_s:
                idx = int(bt * fs)
                if idx - half > 0 and idx + half < n:
                    X_noisy[idx - half: idx + half, ch] += np.hanning(2 * half) * blink_amp
            
            # Scale muscle noise to baseline RMS
            muscle_amp = ch_rms * muscle_multiplier
            ms, me = int(0.4 * fs), int(0.45 * fs)
            if me < n:
                X_noisy[ms:me, ch] += rng.standard_normal(me - ms) * muscle_amp
                
    return X_noisy, applied
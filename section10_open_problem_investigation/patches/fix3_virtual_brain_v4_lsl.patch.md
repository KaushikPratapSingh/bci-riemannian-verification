# Fix #3: Sharpen the alpha oscillator's damping in `virtual_brain_v4_lsl.py`

## Change (line 76)

```python
# BEFORE
osc_alpha = AR2Oscillator(SUBJECT["alpha_peak_hz"] - 0.25,        0.93, noise_std=8.0).warm_up()
```

```python
# AFTER
osc_alpha = AR2Oscillator(SUBJECT["alpha_peak_hz"] - 0.25,        0.99, noise_std=8.0).warm_up()
# Damping changed from 0.93 -> 0.99 (verified fix, see paper Section 9 /
# REPRODUCTION_GUIDE.md). At r=0.93 the alpha resonance has a ~5.8 Hz
# -3dB bandwidth, genuinely bleeding ~19% of its own power into the
# adjacent 13-30 Hz beta band (confirmed by standalone oscillator PSD
# test -- this is real generated signal energy, not a spectral-estimation
# artifact, and is NOT fixed by post-hoc guard-banding or higher-resolution
# Welch windows). r=0.99 gives a ~0.8 Hz bandwidth, consistent with the
# narrow alpha peaks typical of real resting-state EEG, and reduces the
# beta-band leakage to ~2% of alpha's own power.
```

## Why this is the correct fix, not just a parameter tune

The originally reported REST->COGNITIVE beta anomaly (-33.0%, direction
wrong) was attributed in the paper to "AR(2) simulator band-coupling
limitations" as one candidate explanation, without confirming which
mechanism was responsible. Direct measurement on real calibration data
localized the anomaly to channels with high alpha spatial-mixing weight
(Ch3/Ch4: A_MIX alpha column 0.45-0.48) showing large beta decreases that
closely track their large alpha decreases, while low-alpha-weight channels
(Ch1/Ch2: 0.20-0.22) showed the correct beta increase throughout. Standalone
testing of the AR2Oscillator class confirmed this is caused by the alpha
oscillator's damping being too loose (r=0.93), letting real oscillator
energy spill into the beta band.

## Verified result (offline replica of the state-dependent generation +
## A_MIX spatial mixing logic; REST vs WORKLOAD state, seeds 1/2)

| Channel | OLD (r=0.93) % change | NEW (r=0.99) % change |
|---|---|---|
| Ch1 | +20.7% | +44.3% |
| Ch2 | +13.8% | +43.2% |
| Ch3 | -41.7% (wrong direction) | +39.0% (correct) |
| Ch4 | -46.0% (wrong direction) | +39.7% (correct) |

All four channels now show a consistent, correct-direction beta ERS
increase of comparable magnitude, resolving the anomaly.

## What still needs to happen before this goes in the paper

This fix was verified by replicating the AR2Oscillator + A_MIX generation
logic offline (no LSL streaming available in this environment). Before
citing the corrected numbers in the paper, re-run the actual live pipeline
(`virtual_brain_v4_lsl.py` -> `calibration_orchestrator_v2.py` ->
`realtime_inference_engine_v2_lsl.py` -> `evaluate_session_v2.py`) end to
end with this one-line change, on your machine, and confirm the corrected
beta ERD/ERS numbers match this offline projection before updating
Section 6.3 and Table (session results).

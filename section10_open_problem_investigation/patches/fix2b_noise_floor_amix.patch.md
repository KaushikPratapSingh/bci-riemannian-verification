# Fix #2b: Ch1/Ch2 residual out-of-range slope — lever identified, exact
# magnitude requires live calibration (not a confident offline prescription)

## What was tested and ruled out

Your suggestion to investigate the A_MIX spatial mixing weights was the
right instinct, but the actual data does not support the original
explanation ("Ch1/Ch2 have low beta weight, hence steeper slope by
design"). Tested directly:

- Increasing Ch1/Ch2's beta mixing weight 3.5x (0.10/0.12 -> 0.35/0.37):
  slope moved only -3.158 -> -3.121 -- negligible.
- Reducing theta's dominant weight from 0.85/0.80 down to 0.30/0.40 while
  raising beta: slope moved only to -3.05 at best -- still nowhere near
  the -2.5 boundary.

**Conclusion: spatial mixing weights (A_MIX) are not the lever.** The
original caveat's explanation was wrong and should not be used in the
paper.

## What actually works: pink-noise background floor amplitude

The only broadband (non-narrowband-oscillator) element in the signal
chain is `PinkNoiseGenerator`'s `std` parameter (currently 2.5, shared
across all 4 channels via `bg_gens`). Raising it is the correct lever:

| pink_std | Ch1 slope | Ch2 slope | Ch3 slope | Ch4 slope | All in range? | SQI proxy |
|---|---|---|---|---|---|---|
| 2.5 (current) | -3.158 | -3.170 | -3.223 | -3.213 | No | 0.996-0.998 |
| 30 (tested) | -2.410 | -2.388 | -2.354 | -2.353 | **Yes** | 0.979-0.980 |

At pink_std=30, all four channels converge to a consistent, coherent
in-range slope (a more physiologically realistic outcome than the current
setup, where Ch3/Ch4 land in range only by coincidence of their oscillator
mixing ratios). Confirmed this does **not** break the SQI gate (stays at
0.98, well above the 0.95 threshold) in the offline model.

## Why this patch does NOT include a specific recommended pink_std value

Direct comparison against real session data (1784581528) exposes a serious
gap in the offline replica's fidelity: at the *current* pink_std=2.5, the
offline model predicts Ch3/Ch4 slopes of -3.22/-3.21, but the real recorded
session (with the fmax=23 fix alone, no noise-floor change) already
measures -2.15/-2.08 for those same channels -- already in range, no fix
needed. The gap for Ch1/Ch2 is smaller (-3.16 offline vs -2.84 real) but
still substantial.

This means the offline single-state synthetic replica is missing spectral
diversity that the real multi-state session naturally provides (state
transitions, blinks, muscle artifacts, drift) -- likely flattening the
real PSD more than a static single-state synthetic REST block can capture.
**Calibrating pink_std=30 purely against the offline model would very
likely overcorrect the real system by a large margin.**

## Recommendation

1. Do **not** apply pink_std=30 directly based on this offline test alone.
2. Start conservative: try pink_std=6-8 (a ~2.5-3x increase, roughly
   scaled to the smaller real-data gap actually observed for Ch1/Ch2) on
   the live pipeline (`virtual_brain_v4_lsl.py`), re-run
   `evaluate_session_v2.py` with the fmax=23 fix already applied, and
   check the real resulting slope.
3. Iterate empirically from there rather than trusting either extreme
   (2.5 or 30) as precise. This is fundamentally a live-calibration
   problem, not something that can be finalized from offline replication.
4. Ch3/Ch4 may need **no change at all** -- real data shows them already
   in range with just the fmax fix.

## Bottom line for the paper

Report this as: "the residual Ch1/Ch2 out-of-range slope is attributable
to the background broadband noise floor amplitude, not spatial mixing
design (an earlier hypothesis, tested and ruled out) -- confirmed
qualitatively, with precise recalibration deferred to live re-verification
before a specific parameter change is finalized in the codebase." This is
consistent with how Fix #3 was handled (offline-verified direction,
live re-run required to confirm exact magnitude).

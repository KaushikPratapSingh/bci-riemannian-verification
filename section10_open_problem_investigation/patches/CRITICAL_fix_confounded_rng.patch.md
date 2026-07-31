# Critical Fix: Remove the Hidden Confounds That Made Every Prior Comparison Invalid

## The core discovery

`virtual_brain_v4_lsl_fix.py` line 23:
```python
RNG = np.random.default_rng(int(time.time()) % 100000)
```

This seeds the **entire virtual subject's physiological profile** --
`alpha_peak_hz` (8.5-12.5 Hz), `amplitude_scale` (0.7-1.4x, a +/-40% swing
in overall signal amplitude), `beta_reactivity` (0.8-1.6x), `baseline_fatigue`
(0-0.3), and `beta_peak_hz` (18-25 Hz) -- fresh, from wall-clock time, on
**every single run**.

Combined with:
1. The noise generators (`PinkNoiseGenerator`, `AR2Oscillator`) drawing from
   NumPy's global unseeded random state (`np.random.standard_normal`), also
   fresh every run.
2. `--manual` mode requiring a human to hand-time state transitions via
   keyboard, so block durations differ run to run too.

...every comparison made across the pink_std sweeps, the state-scaled
configs, and the REST_STD test was confounded by three uncontrolled sources
of variation simultaneously, on top of the one parameter deliberately being
changed. This is the most likely explanation for results that looked
contradictory (e.g., REST_STD=15 producing a *worse* REST slope than
REST_STD=11 -- fully explainable by landing on a different `amplitude_scale`
or `alpha_peak_hz`, unrelated to REST_STD at all).

## The fix: pin the subject seed

```python
# BEFORE
RNG = np.random.default_rng(int(time.time()) % 100000)

# AFTER
RNG = np.random.default_rng(42)   # any fixed integer -- the value doesn't
                                    # matter, only that it's constant across
                                    # comparison runs
```

## Also pin the noise-generator random stream

The `PinkNoiseGenerator` and `AR2Oscillator` classes call bare
`np.random.standard_normal(...)`, which draws from NumPy's global state --
also unseeded, also different every run. Add this near the top of the
script, before any generator objects are constructed:

```python
np.random.seed(42)   # same or different fixed value from RNG above --
                       # controls the global stream used by
                       # PinkNoiseGenerator.step() and AR2Oscillator.step()
```

## Also remove the manual-timing confound

Run without `--manual` (the autonomous mode already exists in the script --
`state_machine_autonomous()`, lines 231-248) so state transitions follow a
Markov schedule in code rather than human reaction time. Good news: this
function's transitions (`np.random.choice(3, p=row)`) also draw from the
same global `np.random` stream, so the single `np.random.seed(42)` fix above
makes state timing reproducible automatically too -- no separate fix needed
here. Two runs with the same seed will now follow the identical state
sequence and durations, not just the identical noise and subject profile.

## Why this changes everything about how to proceed

With the subject profile, the noise draws, and the state timing all fixed,
**two runs with the same parameter values will now produce identical
output** (a trivial, free sanity check: run the exact same config twice and
confirm the reports match exactly). More importantly, **a single run per
parameter value now becomes a legitimate, controlled comparison** -- because
the only thing differing between a "REST_STD=11" run and a "REST_STD=15"
run is REST_STD itself, not three other hidden variables riding along with
it.

This means we do not need the 10-repeat empirical characterization discussed
earlier. That plan existed specifically to average out noise we now know how
to eliminate directly. One clean run per candidate value, with everything
else pinned, tells us the true causal effect of that parameter.

## Recommended next step

1. Apply both seed fixes above.
2. Run once at the last known configuration (REST=11, COGNITIVE=20,
   MOTOR=28 -- the run that got all three distances passing) to get a
   fixed, reproducible baseline.
3. Run again with only REST_STD changed (e.g., to 15, or another value),
   holding the two seeds and COGNITIVE_STD/MOTOR_STD constant, and this
   time trust the comparison -- any difference is now attributable to
   REST_STD alone, not to a different virtual subject or a different random
   draw.

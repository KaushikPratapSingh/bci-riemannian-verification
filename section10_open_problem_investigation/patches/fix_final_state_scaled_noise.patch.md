# Fix (Recommendations #2 + #3, final iteration): State-Scaled Noise Floor

## The problem this solves

Every flat (state-independent) `pink_std` value tested so far (6, 8, 10, 11,
30) forced a trade-off: raising the noise floor enough to fix the 1/f slope
also diluted the geodesic distances between REST/COGNITIVE/MOTOR states,
because the *same* broadband noise was added identically to all three states,
shrinking their relative covariance contrast. Two independent live runs at
pink_std=11 each failed a different one of the two objectives (slope vs.
Cog<->Motor distance) rather than reliably passing both.

## The fix: make the noise floor state-dependent, not flat

**Literature grounding:** the aperiodic (broadband, non-oscillatory) spectral
*offset* -- overall broadband power level -- is documented to increase with
cortical arousal/activation in real human EEG, not stay flat across states.
This has been shown via ECoG finger-movement tasks (Miller et al. 2009,
*PLoS Comput Biol*), broadband-power-vs-firing-rate correlation (Manning
et al. 2009, *J Neurosci*), and most directly, a 2025 study combining real
EEG (DEAP dataset) with a biophysical corticothalamic model showing the
aperiodic offset rises with arousal, mechanistically traced to increased
thalamic inhibition (Borah, Pathak & Banerjee, 2025, *Imaging Neuroscience*,
doi: 10.1162/imag_a_00451). This directly motivates giving REST the lowest
noise floor and the active states (COGNITIVE, MOTOR) higher, *distinct*
floors, rather than one shared constant.

**Caveat, stated plainly:** the same 2025 study found the spectral *exponent*
(steepness) also increases with arousal in real humans -- a more complex
joint relationship than this fix implements. This patch targets the offset
only, calibrated empirically against this pipeline's own slope/distance
gates, not as a full replication of the cited biophysical mechanism.

## Verified in an offline replica (not yet live-tested)

Using the same generation + A_MIX mixing + windowed Ledoit-Wolf + Riemannian
mean methodology as the live pipeline, three state-scaled configurations
passed slope AND all three pairwise geodesic distances simultaneously for
the first time across this entire investigation:

| rest | cog | motor | slope | d(R,C) | d(R,M) | d(C,M) | All pass? |
|---|---|---|---|---|---|---|---|
| 30 | 60 | 80 | -2.450 | 2.448 | 3.505 | 1.064 | Yes |
| 28 | 55 | 75 | -2.494 | 2.352 | 3.481 | 1.137 | Yes |
| 32 | 50 | 70 | -2.410 | 1.569 | 2.780 | 1.221 | Yes |

The working pattern: COGNITIVE needs roughly **1.6-2.0x** REST's noise floor;
MOTOR needs roughly **2.2-2.7x** REST's floor. Both must be *distinct from
each other*, not just both "higher than REST" -- giving COGNITIVE and MOTOR
the same elevated floor was tested first and collapsed Cog<->Motor to near
zero (0.06-0.09), because it made those two states too similar to each other
even as both separated well from REST.

## Why the offline numbers above are NOT what to put in the live script

This offline replica has consistently required larger absolute noise values
than the real pipeline to hit the same slope target throughout this
investigation (e.g., offline needed pink_std~30 where real recorded sessions
passed around p=8-11 flat). Applying 30/60/80 directly would very likely
overcorrect the real system. What transfers is the **ratio**, not the
absolute magnitude.

**Recommended first live test**, anchoring to REST=11 (already confirmed to
pass the slope gate on its own in real data) and applying the offline-derived
ratios:

```
REST_STD      = 11
COGNITIVE_STD = 20   # ~1.8x REST
MOTOR_STD     = 28   # ~2.5x REST
```

## Code change

```python
# BEFORE (line ~81-92)
class PinkNoiseGenerator:
    def __init__(self, n_octaves=4, std=2.5):
        self.states = np.zeros(n_octaves)
        self.poles  = np.array([0.99, 0.97, 0.93, 0.85])[:n_octaves]
        self.std    = std / n_octaves

    def step(self):
        w            = np.random.standard_normal(len(self.states)) * self.std
        self.states  = self.poles * self.states + w
        return float(np.sum(self.states))

bg_gens = [PinkNoiseGenerator() for _ in range(CHANNELS)]
```

```python
# AFTER
class PinkNoiseGenerator:
    def __init__(self, n_octaves=4, std=2.5):
        self.states = np.zeros(n_octaves)
        self.poles  = np.array([0.99, 0.97, 0.93, 0.85])[:n_octaves]
        self.std    = std / n_octaves

    def step(self):
        w            = np.random.standard_normal(len(self.states)) * self.std
        self.states  = self.poles * self.states + w
        return float(np.sum(self.states))

# State-scaled noise floors (literature-grounded: aperiodic offset increases
# with arousal -- see patch notes for citations and the offline verification
# that motivated these starting values). Three separate generator sets are
# needed since PinkNoiseGenerator's std is fixed at construction time.
REST_STD, COGNITIVE_STD, MOTOR_STD = 11, 20, 28
bg_gens_rest      = [PinkNoiseGenerator(std=REST_STD) for _ in range(CHANNELS)]
bg_gens_cognitive = [PinkNoiseGenerator(std=COGNITIVE_STD) for _ in range(CHANNELS)]
bg_gens_motor     = [PinkNoiseGenerator(std=MOTOR_STD) for _ in range(CHANNELS)]

def bg_gens_for_state(state):
    if state == "REST":
        return bg_gens_rest
    elif state == "WORKLOAD":
        return bg_gens_cognitive
    else:
        return bg_gens_motor
```

```python
# In the main loop, BEFORE (line ~321):
bg      = np.array([g.step() for g in bg_gens])

# AFTER:
bg      = np.array([g.step() for g in bg_gens_for_state(state)])
```

## What to send back for verification

A fresh live run at these settings, ideally 2-3 repeats given the run-to-run
variance already documented in this investigation (see prior p=11 analysis)
-- one clean run is promising but not yet sufficient evidence given how
noisy these metrics have shown themselves to be at close-to-boundary values.

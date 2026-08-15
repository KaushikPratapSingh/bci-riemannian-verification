# Open Problem: The 1/f Slope vs. State-Separability Trade-off in the AR(2) Simulator's Broadband Noise Floor

*Compiled from a verification investigation to (a) correct the causal-filter group-delay distortion
of the fitted 1/f spectral slope, and (b) resolve the beta ERD/ERS direction
anomaly. Documented here as an open problem for follow-up work rather than a
closed fix, in keeping with this paper's disclosure-over-omission principle
(Section 9).*

**Every script, patch, and real live-run evidence file (npz/CSV/PNG)
referenced below is in [`section10_open_problem_investigation/`](./section10_open_problem_investigation/)
— see that folder's README for an index.**

## 1. Summary for a reader in a hurry

Two independently-motivated diagnostic gates in this pipeline's synthetic
validation session --- a physiologically plausible 1/f spectral slope
(target: −1.0 to −2.5), and Riemannian geodesic separability between REST,
COGNITIVE, and MOTOR states (target: all three pairwise distances > 1.0
geodesic unit) --- cannot currently be satisfied simultaneously by adjusting
a single global noise parameter. Raising the simulator's broadband noise
floor enough to correct the spectral slope systematically erodes the
covariance contrast between states, and vice versa. A literature-grounded,
state-dependent noise model (Section 4) got closer than any flat parameter
choice, and one run passed every gate simultaneously, but repeated,
properly-controlled testing (Section 5) shows the fix is not yet reliable
across runs. We document the full trade-off surface, the mechanism, and the
specific next experiments needed, so that this is a resumable problem rather
than a dead end.

## 2. Background: what these two gates measure and why they're both here

The 1/f slope gate (Section 6.3 of the main paper) checks that the
simulator's power spectral density falls off with frequency at a rate
consistent with real human EEG (Klimesch 1999; Nunez & Srinivasan 2006). The
geodesic separability gates check that the Riemannian covariance structure
of REST, COGNITIVE, and MOTOR states is genuinely distinguishable --- the
same property the paper's Cohen's d = 4.967 result (Section 6.3) and the
STEW cross-validation (Section 7) depend on elsewhere. Both gates are
legitimate, independent checks on simulator realism; neither should be
weakened to satisfy the other without a documented reason.

## 3. The originally reported problem and its real cause (resolved)

The paper originally attributed the steep 1/f slope (~−3.4, well outside the
physiological range) to causal-filter group-delay distortion, and the beta
ERD/ERS sign anomaly to unspecified "AR(2) simulator band-coupling
limitations." Neither diagnosis survived direct testing:

- **Group delay was not the cause.** Replacing the causal `lfilter` with
  zero-phase `filtfilt` did not improve the slope (in fact made it
  marginally worse, since `filtfilt` squares the magnitude rolloff near the
  filter's 45 Hz corner). The real issue was the slope-fitting window
  (originally 4-40 Hz) extending too close to that corner, where the
  filter's own rolloff --- not physiology --- dominates the fit. Narrowing
  the fit to 4-23 Hz resolved this specific artifact independent of any
  noise-floor changes (verified against real session data, `fmax` scan
  documented in `recommendation_fixes.zip`).
- **Cross-frequency coupling was not the cause of the beta anomaly.** The
  alpha oscillator's damping (`r=0.93`) gives a ~5.8 Hz resonance bandwidth,
  and standalone testing confirmed ~19% of the alpha oscillator's own power
  genuinely bleeds into the 13-30 Hz beta band at that damping value ---
  real generated signal energy, not a measurement artifact. Sharpening to
  `r=0.99` (physiologically realistic ~0.8 Hz alpha bandwidth) resolved the
  beta ERD/ERS sign error cleanly in an offline replica of the full
  generation+mixing pipeline.

Both of these fixes are narrow, mechanistically understood, and did not
introduce the trade-off described below. The trade-off emerged specifically
from the *residual* slope correction still needed after these two fixes ---
addressed by raising the broadband noise floor, which is where the coupling
to state-separability appears.

## 4. The trade-off: flat noise floor vs. state separability

With the filter-corner and alpha-damping fixes in place, per-channel slopes
on real recorded session data still ran steeper than target (Ch1/Ch2 around
−2.8 to −2.9). The obvious next lever --- raising the simulator's background
`PinkNoiseGenerator` amplitude (`pink_std`) --- was tested across a wide
range on live hardware-in-the-loop-free sessions:

| pink_std | slope | in range? | Rest↔Cog | Rest↔Motor | Cog↔Motor | all pass? |
|---|---|---|---|---|---|---|
| 6 | −3.105 | No | 1.149 | 1.671 | 1.275 | No (slope) |
| 8 | −2.612 | No | 1.244 | 1.654 | 1.158 | No (slope) |
| 10 | −3.355 | No | 1.150 | 1.519 | 1.134 | No (slope) |
| 11 (run 1) | −2.440 | **Yes** | 1.111 | 1.385 | 0.906 | No (Cog↔Motor) |
| 11 (run 2) | −2.960 | No | 1.057 | 1.440 | 1.087 | No (slope) |
| 30 | −2.481 | **Yes** | 0.363 | 0.578 | 0.528 | No (all 3 distances) |

The pattern is a genuine mechanistic trade-off, not incidental noise: adding
identical broadband noise to all three states dilutes their covariance
contrast, because the added noise term is common to all of them. At high
enough magnitude (pink_std=30) this is severe and unambiguous; at
intermediate values (pink_std=11) it manifests as marginal, run-to-run
variable failures.

**A confound that inflated the apparent inconsistency in this table:** until
this was identified, `virtual_brain_v4_lsl.py` seeded both its random
subject-profile generator (`alpha_peak_hz`, `amplitude_scale` ∈ [0.7, 1.4],
`beta_reactivity`, `baseline_fatigue`, `beta_peak_hz` — all resampled per
run) and its noise-generator random stream from wall-clock time. Every row
above therefore differs not only in `pink_std` but in a freshly and
independently randomized simulated subject and noise realization. This
likely explains non-monotonic entries in the table (e.g., pink_std=10 giving
a worse slope than both its neighbors). **This has since been fixed** by
pinning both random streams to a constant seed, which should be applied
before any further parameter search (see Section 6).

## 5. The state-dependent noise floor: a literature-grounded partial fix

**Mechanism and justification.** Rather than add identical noise to all
three states, the aperiodic (broadband, non-oscillatory) component of real
human EEG power spectra is documented to scale with cortical
arousal/activation, not stay constant. This is supported by three
independent lines of evidence: ECoG broadband power tracking finger
movement (Miller et al. 2009, *PLoS Comput Biol*, doi:
10.1371/journal.pcbi.1000609), broadband power correlating with local
neuronal firing rate (Manning et al. 2009, *J Neurosci*, doi:
10.1523/JNEUROSCI.2041-09.2009), and a 2025 study combining real human EEG
(DEAP dataset) with a biophysical corticothalamic model showing the
aperiodic *offset* rises with arousal, mechanistically linked to increased
thalamic inhibition (Borah, Pathak & Banerjee, 2025, *Imaging Neuroscience*,
doi: 10.1162/imag_a_00451). This motivates a REST < COGNITIVE < MOTOR noise
floor ordering rather than a flat one.

**Caveat stated plainly:** the cited 2025 study also found the spectral
*exponent* (steepness, not just offset) increases with arousal in real
humans — a more complex joint relationship than this fix implements. This
patch targets the offset only, calibrated empirically against this
pipeline's own gates, not as a full replication of the cited mechanism.

**Offline verification (pre-seed-fix).** Using a from-scratch replica of the
generation, A_MIX spatial mixing, windowed Ledoit-Wolf covariance, and
Riemannian Fréchet mean pipeline, three state-scaled configurations (with
COGNITIVE ≈ 1.6-2.0× REST's floor and MOTOR ≈ 2.2-2.7× REST's floor, the two
active states kept *distinct from each other*, not just both elevated)
passed every gate simultaneously in simulation:

| rest | cog | motor | slope | d(R,C) | d(R,M) | d(C,M) | all pass? |
|---|---|---|---|---|---|---|---|
| 30 | 60 | 80 | −2.450 | 2.448 | 3.505 | 1.064 | Yes |
| 28 | 55 | 75 | −2.494 | 2.352 | 3.481 | 1.137 | Yes |
| 32 | 50 | 70 | −2.410 | 1.569 | 2.780 | 1.221 | Yes |

**Live verification (still pre-seed-fix at this point):** a single live run
at `REST_STD=11, COGNITIVE_STD=20, MOTOR_STD=28` reproduced this result on
real (not offline-replica) data — the first and only time in this
investigation that all four gates (slope + 3 distances) passed on a live
run:

```
slope = -2.850 (aggregate; per-state: REST -2.718 fail, COGNITIVE -2.464 pass, MOTOR -2.083 pass)
Rest↔Cog   = 1.334   (pass)
Rest↔Motor = 2.136   (pass)
Cog↔Motor  = 1.017   (pass)
```

A follow-up attempt to fix the one remaining failure (REST's own slope and
alpha-prominence) by raising `REST_STD` alone from 11 to 15 made
*everything* worse simultaneously, including COGNITIVE's slope, which had
not been touched. This was the signal that led to identifying the RNG
confound described in Section 4 — a "controlled" single-parameter change
was not actually controlled.

## 6. Post-seed-fix results: the confound explains some, not all, of the instability

After pinning both random streams (see patch `CRITICAL_fix_confounded_rng.patch.md`
in the accompanying materials), two further live runs were completed under
genuinely controlled conditions (same seed, same simulated subject):

```
Run A: Rest↔Cog = 0.7434   Rest↔Motor = 1.4007   Cog↔Motor = 0.9189
Run B: Rest↔Cog = 0.7283   Rest↔Motor = 1.4204   Cog↔Motor = 0.9775
```

These two runs are close to each other (within ~0.02-0.06 units per pair) —
consistent with the seed fix working as intended, and consistent with each
other in showing **both Rest↔Cog and Cog↔Motor failing**, with only
Rest↔Motor reliably passing. This is a different, and now reproducible,
failure pattern from the earlier (confounded) `REST=11/COG=20/MOTOR=28` run
that passed everything. The exact REST/COGNITIVE/MOTOR values used for runs
A and B were not recorded before this investigation paused; recovering them
(from terminal output or script state at the time) is the single highest-value
piece of missing information for whoever picks this up next.

## 7. What we can respons­ibly claim, and what remains open

**Can claim:**
- The originally-reported causes of both the slope and beta-anomaly
  findings were wrong; the real mechanisms (filter-corner contamination,
  alpha oscillator bandwidth) are identified and independently fixed.
- A flat, state-independent noise floor cannot satisfy both the slope and
  full separability gates simultaneously at any tested value — this is a
  genuine structural property of the current architecture, not a tuning
  failure.
- A state-dependent noise floor, motivated by real EEG literature on
  arousal-linked broadband power, is the right *class* of fix — it achieved
  a full simultaneous pass at least once, which a flat floor never did
  across seven tested values.
- A significant, previously-unidentified confound (time-seeded subject
  profile and noise generators) inflated the apparent inconsistency of
  early results and has been fixed going forward.

**Cannot yet claim:**
- That any specific state-scaled configuration reliably passes all gates.
  Two seed-controlled runs since the fix both show a Rest↔Cog / Cog↔Motor
  failure pattern; only one (pre-fix, confounded) run passed everything.
- A quantitative model relating REST/COGNITIVE/MOTOR noise-floor ratios to
  gate outcomes — the offline replica and the live pipeline have shown
  meaningfully different absolute magnitudes throughout this investigation
  (the offline replica consistently requires larger noise values to hit the
  same slope target), so offline parameter search cannot fully substitute
  for live verification.

## 8. Concrete next steps for a follow-up researcher

1. **Recover or re-derive the exact REST_STD/COGNITIVE_STD/MOTOR_STD used in
   the two post-seed-fix runs (Section 6)**, and re-run once with those
   values logged explicitly (e.g., printed to the terminal banner at
   startup, as `SUBJECT` already is) so results are self-documenting going
   forward.
2. **With the RNG now pinned, a single controlled run per candidate
   configuration is sufficient** — the earlier plan to average over ~10
   repeats to characterize a "pass rate" is no longer necessary, since the
   dominant source of run-to-run inconsistency (the confound) is resolved.
   A small grid search (e.g., 3×3 over COGNITIVE and MOTOR multipliers
   relative to a fixed REST value) with the seed pinned should be
   sufficient to map the real trade-off surface.
3. **Investigate whether COGNITIVE and MOTOR need a larger enforced gap.**
   Every failure mode observed post-seed-fix and in the original
   same-floor test (Section 4 preamble) implicates COGNITIVE and MOTOR
   converging toward each other specifically, more than either converging
   toward REST. A worthwhile experiment: hold REST and MOTOR fixed at
   known-good values and sweep COGNITIVE alone across a wide range,
   isolating which specific direction is fragile.
4. **The NaN SQI bug reported in one of the post-seed-fix runs remains
   uninvestigated** — its source is upstream in `realtime_inference_engine_v2_lsl.py`
   (which writes the `SQI` column that `evaluate_session_v2.py` only reads),
   not in either script modified during this investigation. Should be
   triaged independently of the noise-floor question, since it may or may
   not be related.

## 9. Suggested framing for the paper

Given the time cost of exhaustively resolving this live, we recommend
presenting this as an explicitly flagged **open problem** in the
Limitations section, structured the same way as the paper's other disclosed
findings (the Subject-5 bug, the swelling non-monotonicity retraction): state
what was found, what was ruled out, what partial progress was made, and what
a full resolution would require. This is consistent with the paper's own
stated principle that "the process of finding and fixing bugs is itself a
publishable contribution when documented with sufficient transparency to be
reproducible" (Section 9) — an honestly incomplete but rigorously
characterized negative result is more valuable to the field than a
cherry-picked single passing run presented as a clean fix.

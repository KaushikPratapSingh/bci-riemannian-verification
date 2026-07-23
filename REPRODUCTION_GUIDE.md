# Reproduction Guide
## Leakage-Audited Riemannian Classification and Guardrail-Constrained AI Narration for Low-Channel EEG

> **Changelog (this verification pass):**
> - §1.2 corrected: the script for the synthetic FastICA/SOBI benchmark
>   (+19.99 dB, 51.6%/43.2%) is `heuristic_sensitivity.py`, not
>   `phase1_diagnostic.py`. The latter produces an unrelated result (§2.5,
>   real-PhysioNet matched-reference diagnostic). Verified independently
>   against `Terminal_Outputs.txt`.
> - `phase1_diagnostic.py` and `phase_i_diagnostic.py` relocated in the
>   repository structure (§6) from Stage 1 to Stage 2, matching what they
>   actually compute.
> - §1.9 and §1.10 added: `proper_stats.py` and `phase2_sqi.py` were both
>   referenced elsewhere in this guide/paper but never given a runnable step.
>   Both now have one.

Every verified number in the paper traces to one of the scripts below, using their
**actual filenames as they exist in this repository** — not idealised placeholder
names. Run top-to-bottom to reproduce every claim before submission.

---

## 0. Environment Setup (do this once)

```bash
python -m venv bci_env
source bci_env/bin/activate        # Linux/macOS
# bci_env\Scripts\activate         # Windows

pip install numpy scipy scikit-learn pandas matplotlib mne pylsl websockets

python -c "import numpy, scipy, sklearn, mne; print('Environment OK')"
```

---

## 1. Stage 1 — Standalone Verifications (no external data needed, run in <5 min)

### 1.1 SOBI Known-Answer Sanity Check
**Claim:** SOBI recovers two mixed sinusoids at correlation = 1.0000 for both components.
```bash
py sobi.py
```
If you see values below 0.95, you've reproduced the pre-fix bug (Jacobi-rotation
sign error, Bug #5). The corrected implementation passes at 1.0000.

### 1.2 FastICA vs. SOBI Single-Run Benchmark
**Claims:** Pre-pipeline SNR −18.68 dB; FastICA post-SNR +1.31 dB (gain +19.99 dB),
correlation 51.6% [49.1%, 54.1%]; SOBI +0.23 dB gain, 43.2% [40.7%, 45.7%].
```bash
py heuristic_sensitivity.py
```
**Correction (found during repo verification, not previously documented):** an
earlier version of this guide listed `phase1_diagnostic.py` here. That was
wrong — `phase1_diagnostic.py` produces a different result entirely (the
real-PhysioNet matched-reference diagnostic in §2.5 below, −0.34 dB / 41.4%).
Confirmed against `Terminal_Outputs.txt`: `py heuristic_sensitivity.py` is
immediately followed by the 51.6%/43.2% numbers above, and its source builds
the exact synthetic theta/alpha/beta three-source mixture this claim
describes. If your local copy of this guide still says `phase1_diagnostic.py`
for this step, it predates this correction.

### 1.3 FastICA vs. SOBI Paired Statistical Test (60 trials)
**Claim:** Wilcoxon W=962, p=0.729 — no statistically significant difference.
```bash
py bootstrap_and_permutation.py
```

### 1.4 AR(4) Harder Benchmark (band-limited stochastic sources)
**Claim:** FastICA 54.7% vs. SOBI 54.3% (gap <1 pp) — confirms 1.3's null result
on a more realistic signal model than pure sinusoids.
```bash
py ar4_benchmark.py
```

### 1.5 MiniEEGNet Gradient Check
**Claim:** Max relative gradient error = 4.35×10⁻¹⁰ (threshold 10⁻³, PASS).
```bash
py mini_eegnet.py
```
This is a central-difference check with step h=10⁻⁵; the theoretical agreement
floor is O(h²)=10⁻¹⁰, so this result confirms the analytical gradient is correct
*to the limit of what the finite-difference method itself can resolve* — see
Section 6.2 of the paper for the full error-budget discussion. If max error = 1.00
exactly, you've reproduced the pre-fix harness bug (index transposition, Bug #6).

### 1.6 Riemannian Geometry Sanity Check
**Claim:** max|computed Fréchet mean − true mean| ≈ 8.88×10⁻¹⁵ – 2.22×10⁻¹⁴.
```bash
py riemannian.py
```
This is the expected float64 round-off accumulation for a well-conditioned 4×4
SPD matrix over 1–2 solver iterations — see Section 6.2 for why this specific
magnitude, not just its smallness, is the correct diagnostic signature.

### 1.7 OLA Streaming Pipeline Benchmark
**Claims:** mean processing time 0.041–0.085 ms/step (occasional transients to
~1.3 ms), ≥35× headroom under the 50 ms gate; streaming-vs-batch correlation
98.2% (14.24 dB); streaming/batch vs. ground truth 42.8%/47.5%.
```bash
py phase3_streaming_ola.py
```

### 1.8 Four-Way Regression Tournament (Figure 6)
**Claims:** Ridge R²=0.649, RF R²=0.815, MiniEEGNet R²=0.865, Riemannian+Ridge
R²=0.877; training times 0.001s/0.066s/5.052s/0.006s respectively.
```bash
py phase4_tournament.py
py four_way_tournament.py     # Riemannian+Ridge row specifically
```

### 1.9 Section 4.4 FastICA-vs-SOBI Benchmark (canonical script)
**Gap found during repo verification, filled here:** §2.2 above references
`proper_stats.py` by name as "the Section 4.4 synthetic FastICA/SOBI
benchmark" script, to distinguish it from `loso_significance_test.py` — but
this guide never previously gave it a runnable step. Adding one now:
```bash
py proper_stats.py
```
This combines the FastICA/SOBI comparison with the MiniEEGNet and Riemannian+Ridge
tournament in one script (it imports from `mini_eegnet.py`, `phase4_tournament.py`,
and `sobi.py`). Cross-check its FastICA/SOBI output against 1.2's
`heuristic_sensitivity.py` numbers above — they should agree, since both draw
on the same synthetic benchmark; if they diverge, that's worth flagging before
submission rather than treating either run as more authoritative than the
other.

### 1.10 SQI / Lead-Off Detachment Validation
**Claim (stated narratively in paper §4.3, not previously given a script
here):** synthetic lead-off event at t=6.0s; SQI drops below 0.20 within
100ms of detachment (gate target: <200ms).
```bash
py phase2_sqi.py
```
This is the standalone synthetic validation backing the online `SQI < 0.95`
exclusion gate described in §4.3 — distinct from the real-session SQI mean
(0.9974) reported in Stage 3 §3.1, which is a different computation over real
data, not this synthetic detachment test.

---

## 2. Stage 2 — PhysioNet BCI2000 Verifications

### 2.0 Download PhysioNet Data
```bash
python -c "
from mne.datasets import eegbci
paths = eegbci.load_data(subjects=[1], runs=[3], path='./data/physionet/', verbose=False)
print('Subject 1 downloaded:', paths[0])
"
```
> **Common error:** `TypeError: load_data() got an unexpected keyword argument 'subject'`.
> The correct keyword in current MNE versions is `subjects` (plural, list-valued), not `subject`. See `phase1_physionet_validation.py` for the corrected call.

### 2.1 ML Tournament Version History — v10 → v34 (the 35-version gauntlet)

**Use only `ml_cohort_tournament_v34.py`, cleaned.** Do not use `v34.1` — confirmed
duplicate of v34, see §2.4 below.

#### v10 — TSA Leakage Bug (superseded, kept for the case study)
```bash
py ml_cohort_tournament_v10.py
```
Fits the TSA projector on REST-only trials. On pure Gaussian noise input, this
produces **76.20% accuracy** — proof the "signal" is a geometric artefact of
asymmetric fitting, not biology. This is Figure 4 (left panel) in the paper.

#### v11 — TSA Leakage Fixed
```bash
py ml_cohort_tournament_v11.py
```
TSA fitted globally on all training trials (REST + MOTOR combined). Same pure-noise
input now collapses to **51.4%** (chance). Figure 4 (right panel).

#### v29 → v30 — Silent Exclusion Bug Found and Fixed
Documented via terminal-output arithmetic cross-check (reported mean did not match
the mean of logged per-subject values). Fixed in v30: every processed subject
always contributes to the primary mean.

#### v34 — Final canonical version (cleaned)
```bash
py ml_cohort_tournament_v34.py > ML_Cohort_v34_final.txt
py ml_cohort_tournament_v34.py --run-negative-control >> ML_Cohort_v34_final.txt
```
**Before running:** confirm your copy of `ml_cohort_tournament_v34.py` does **not**
contain the following block (search for `sid == 5`); if it does, remove it —
this was an undisclosed synthetic-noise injection affecting Subject 5 only,
found during paper verification (Bug #7):
```python
# REMOVE if present:
if sid == 5:
    trial_raw = trial_raw.copy() + np.random.randn(*trial_raw.shape) * 350.0
```

**Expected output (corrected script):**
```
Model               Intact (n=50)   Shuffled baseline   Real gap
RF Raw                  66.14%           50.39%          +15.75 pp
RF on TSA Features      74.72%           47.88%          +26.84 pp
SVM Raw                 60.39%           49.69%          +10.70 pp
Gated SVM on TSA        71.96%           49.11%          +22.85 pp
```
This is Figure 5 (t-SNE) and the Section 5.3 table in the paper.

### 2.2 LOSO Significance Test
**Claim:** Wilcoxon signed-rank W=66.0, p=1.36×10⁻⁷ (n=50 paired subjects,
42/50 higher under intact labels).

This uses a **general-purpose, version-agnostic** script — not `proper_stats.py`,
which is specific to the Section 4.4 synthetic FastICA/SOBI benchmark and does
not operate on LOSO fold accuracies at all.
```bash
py loso_significance_test.py --intact ML_Cohort_v34_final.txt --shuffled ML_Cohort_v34_final.txt --metric TSA
```
Passing the same combined log to both `--intact` and `--shuffled` is correct —
the script automatically finds the two separate `RUNNING NESTED LEAVE-ONE-SUBJECT-OUT`
blocks (one per run) inside it. To compare two separate log files instead, point
each flag at its own file. See `--help` for the full usage note.

**Expected output:**
```
Metric: TSA  |  n subjects paired: 50
Intact   mean: 74.72%
Shuffled mean: 47.87%
Mean paired difference (intact - shuffled): 26.85 pp
95% bootstrap CI on mean difference: [20.42, 33.16] pp
Sign test: intact > shuffled for 42/50 subjects, < for 5/50, tied for 3/50
Wilcoxon signed-rank test (paired, two-sided): W=66.0, p=0.000000 -> significant
```
(The `p=0.000000` is print rounding; the underlying value is 1.36×10⁻⁷.)

### 2.3 Subject Normalisation Study
**Claims:** 13.1× EI variability across 50 subjects (6.2× excluding outliers);
0.84 SD mean generalisation penalty; 9.64 SD worst-case error under global baseline.
```bash
py subject_normalization_pipeline.py --data_dir ./data/physionet/
```

### 2.4 v34 vs. v34.1 — Duplicate Confirmation (Bug #8)
If you have both `ml_cohort_tournament_v34.py` and `ml_cohort_tournament_v34_1.py`
in your working folder, confirm they are duplicates before treating either as
canonical:
```bash
diff ml_cohort_tournament_v34.py ml_cohort_tournament_v34_1.py
```
Every remaining diff line should be a version-label string, a `print()` banner,
or a comment — none should touch computation. If both contain the Subject 5
injection, fix both the same way as §2.1 before comparing. Once cleaned, running
either script produces **identical** results (confirmed independently on the
author's machine) since both use `random_state=42` throughout (RF, SVM, inner
CV splits, and the t-SNE projection). **Retain only v34** going forward.

### 2.5 Phase 1 PhysioNet Diagnostic (Figure 8)
**Claims:** Against raw unfiltered reference: SNR +16.71 dB, correlation 22.1%
(both fail gate). Against similarly-filtered reference: SNR −0.34 dB, correlation
41.4% (passes r>40% gate).
```bash
py phase1_physionet_validation.py
py phase_i_diagnostic.py
```

---

## 3. Stage 3 — 4-Channel Closed-Loop Simulation

### 3.1 Calibration and Real-Time Session (4 terminal windows)

**Terminal 1 — start the simulator in manual mode:**
```bash
py virtual_brain_v4_lsl.py --manual
# Type 1 (REST), 2 (COGNITIVE), or 3 (MOTOR), then Enter, to set the state
```

**Terminal 2 — run the 15-minute calibration protocol:**
```bash
py calibration_orchestrator_v2.py
# Prompts you to press Enter to begin each 4-minute block.
# Set the matching state in Terminal 1 BEFORE pressing Enter in Terminal 2.
# Outputs: calibration_resting_alpha.csv, calibration_cognitive_load.csv,
#          calibration_motor_imagery.csv, structural_brain_baseline.npz
```

**Stop the manual simulator (Ctrl+C in Terminal 1), then restart it autonomously:**
```bash
py virtual_brain_v4_lsl.py
```

**Terminal 3 — run real-time inference:**
```bash
py realtime_inference_engine_v2_lsl.py
# Loads baselines from structural_brain_baseline.npz automatically.
# Note the printed session timestamp — you need it for evaluation.
# Outputs: eeg_signals_<session>.csv, eeg_metrics_<session>.csv
```

**Stop after ~7+ minutes (Ctrl+C), then evaluate:**
```bash
py evaluate_session_v2.py
# To reproduce this paper's exact numbers on the existing logged session,
# point it at the existing CSVs rather than regenerating a new session —
# check the script's --session_ts argument.
```

**Expected output (session 1784410860, as reported in the paper):**
```
Session:       2,153 windows  (431s = 7.2 min)
SQI mean:      0.9974  [0.9972, 0.9976]
Cohen's d:     4.967   (engaged vs. resting)
Mann-Whitney:  <1e-10  (see paper §6.3 — not independently interpretable due to
                        ~90% sample overlap between consecutive windows)
REST<->COG:    1.2849 geodesic units
REST<->MOTOR:  1.9601 geodesic units
COG<->MOTOR:   1.4095 geodesic units
Alpha desync REST->COG:    -40.5%
Alpha desync REST->MOTOR:  -70.0%
Beta desync REST->COG:     -33.0%  (direction incorrect, expected positive — disclosed, unresolved, see §6.3)
1/f slope:     -3.42  (causal IIR filter artefact, see §6.3 — not a bug)
Figure saved:  bci_evaluation_v2.png  (Figure 7 in the paper)
```

> **Note on session numbers:** `virtual_brain_v4_lsl.py` generates a fresh
> timestamp on every run. To reproduce the *exact* numbers above rather than a
> new, statistically similar session, use the already-logged
> `eeg_signals_1784410860.csv` / `eeg_metrics_1784410860.csv` files included in
> this repository's `data/` folder, rather than regenerating from scratch.

---

## 4. Stage 4 — STEW Dataset Verifications

### 4.0 Download STEW Dataset
```
1. Create a free account at: https://ieee-dataport.org
2. Download: STEW Dataset.zip (~45 MB)
3. Place at: ./data/stew/STEW_Dataset.zip
```

### 4.1 STEW Spectral Analysis
**Claims:** REST 1/f slope −1.878 ± 0.395, 95% CI [−2.607, −1.030]; 41/48 subjects
within human range; alpha prominence 1.70× at 10.25 Hz; per-subject alpha ERD
p=0.940 (not significant — pooled −33% is not a confirmed per-subject finding).
```bash
py process_stew_benchmark.py --zip ./data/stew/STEW_Dataset.zip
```

### 4.2 STEW Riemannian Swelling Sensitivity Sweep (Table II)
**Claims:** Determinant swelling 360×–92,898× across window counts; Riemannian
volume preservation 1.000× in all conditions; Fréchet mean convergence in
19–42 iterations.
```bash
py run_route_a_evaluation.py
```

### 4.3 Multi-Seed Reproducibility Sweep (the swelling retraction)
**Claim:** A 20-seed random-resampling sweep shows the originally-reported
N=1,000 "dip" does not survive averaging over window selection — see paper
Section 9, finding #3, for the full retraction and what the sweep does confirm
instead (monotonically shrinking seed-to-seed variance as N grows).
```bash
py swelling_reproducibility_sweep.py
```
**Resolved (author-confirmed):** this is the canonical script for the 20-seed
sweep. An earlier version of this guide flagged an ambiguity with
`heuristic_sensitivity.py` (which was mistakenly also present in `stage4/`);
that script is the canonical source for the §1.2 synthetic FastICA/SOBI
benchmark only, has no role in the swelling sweep, and the duplicate copy has
been removed from `stage4/`.
Output written to `swelling_reproducibility_results.csv`.

---

## 5. AI Guardrail Verification

**Claim:** 20/20 adversarial prompt categories structurally unreachable by the
constrained decoding grammar — a proof of unreachability given the grammar's
construction, not a statistical evaluation of any specific model (no unaligned
model is queried).

The 20-prompt suite and its structural-coverage argument are documented in full
in Section 8.4 of the paper. There is no standalone script to run for this
section — the guarantee is verified by inspection of the CFG grammar defined in
`llm_context_generator.py` against each of the 20 documented prompt/category
pairs, not by executing a model.

---

## 6. GitHub Repository Structure

```
.
├── README.md
├── REPRODUCTION_GUIDE.md            <- this file
├── requirements.txt
├── paper/
│   └── Paper5_FINAL.md
├── figures/
│   ├── Figure1_preprocessing_chain.png
│   ├── Figure2_FastICA_pipeline.png
│   ├── Figure3_FastICA_vs_SOBI.png
│   ├── Figure4_leakage_tsne.png
│   ├── Figure5_final_cohort_tsne.png
│   ├── Figure6_four_way_tournament.png
│   ├── Figure7_session_dashboard.png
│   └── Figure8_PhysioNet_diagnostic.png
├── stage1_signal_processing/
│   ├── sobi.py  riemannian.py  mini_eegnet.py
│   ├── phase2_sqi.py  phase3_streaming_ola.py
│   ├── phase4_tournament.py  four_way_tournament.py  phase4_noise_sweep.py
│   ├── ar4_benchmark.py  rms_scaled_injection.py
│   ├── bootstrap_and_permutation.py  heuristic_sensitivity.py  proper_stats.py
├── stage2_bci2000_validation/
│   ├── phase1_physionet_validation.py  phase1_diagnostic.py  phase_i_diagnostic.py
│   ├── batch_physionet_validation.py
│   ├── ml_cohort_tournament_v10.py … ml_cohort_tournament_v34.py
│   ├── subject_normalization_pipeline.py
│   └── logs/  ML_Cohort_v10.txt … ML_Cohort_v34.txt
├── stage3_closed_loop_simulation/
│   ├── virtual_brain_v4_lsl.py  calibration_orchestrator_v2.py
│   ├── realtime_inference_engine_v2_lsl.py  evaluate_session_v2.py
│   ├── lsl_real_time_receiver.py  live_receiver_pipeline.py
│   └── phase5_calibration_corrected.py
├── stage4_stew_crossvalidation/
│   ├── process_stew_benchmark.py  run_route_a_evaluation.py
│   └── swelling_reproducibility_sweep.py
├── verification_scripts/
│   └── loso_significance_test.py
└── data/
    ├── calibration_resting_alpha.csv  calibration_cognitive_load.csv  calibration_motor_imagery.csv
    ├── structural_brain_baseline.npz
    ├── eeg_signals_1784410860.csv  eeg_metrics_1784410860.csv
    └── swelling_reproducibility_results.csv
```

### .gitignore
```
data/*.csv
data/*.npz
bci_env/
__pycache__/
*.pyc
results/
```

---

## 7. Verified Numbers Quick-Reference Table

| # | Claim | Script | Expected | Runtime |
|---|---|---|---|---|
| 1 | SOBI sanity check | `sobi.py` | corr = 1.0000 | <5 s |
| 2 | FastICA SNR gain | `heuristic_sensitivity.py` | +19.99 dB | <30 s |
| 3 | FastICA correlation | same | 51.6% [49.1, 54.1] | <30 s |
| 4 | SOBI correlation | same | 43.2% [40.7, 45.7] | <30 s |
| 5 | BSS paired test (60 trials) | `bootstrap_and_permutation.py` | p=0.729, W=962 | ~3 min |
| 6 | AR(4) harder benchmark | `ar4_benchmark.py` | 54.7% vs 54.3%, gap <1pp | ~3 min |
| 7 | Gradient check | `mini_eegnet.py` | 4.35×10⁻¹⁰ | <10 s |
| 8 | Riemannian sanity | `riemannian.py` | 8.88×10⁻¹⁵–2.22×10⁻¹⁴ | <5 s |
| 9 | OLA timing | `phase3_streaming_ola.py` | 0.041–0.085 ms mean | <60 s |
| 10 | Four-way tournament | `phase4_tournament.py`, `four_way_tournament.py` | R²: 0.649/0.815/0.865/0.877 | ~2 min |
| 11 | TSA leakage (v10) | `ml_cohort_tournament_v10.py` | 76.2% on pure noise | 30–90 min |
| 12 | TSA fixed (v11) | `ml_cohort_tournament_v11.py` | 51.4% on pure noise | 30–90 min |
| 13 | RF TSA accuracy (v34, cleaned) | `ml_cohort_tournament_v34.py` | 74.72% | 30–90 min |
| 14 | Shuffled baseline (v34, cleaned) | same, `--run-negative-control` | 47.88% | included |
| 15 | LOSO Wilcoxon significance | `loso_significance_test.py` | W=66.0, p=1.36×10⁻⁷ | <5 s |
| 16 | EI variability | `subject_normalization_pipeline.py` | 13.1× | ~5 min |
| 17 | STEW 1/f slope | `process_stew_benchmark.py` | −1.878 [−2.607, −1.030] | ~2 min |
| 18 | STEW alpha ERD (per-subject, null) | same | p=0.940 | included |
| 19 | Swelling range | `run_route_a_evaluation.py` | 360×–92,898× | 5–15 min |
| 20 | Riemannian volume | same | 1.000× all N | included |
| 21 | Session SQI (1784410860) | `evaluate_session_v2.py` | 0.9974 | real-time |
| 22 | Cohen's d (1784410860) | same | 4.967 | included |
| 23 | Session geodesics | same | 1.2849 / 1.9601 / 1.4095 | included |
| 24 | Phase 1 PhysioNet diagnostic | `phase1_physionet_validation.py` | +16.71/−0.34 dB, 22.1%/41.4% | ~5 min |

---

## 8. Known Deviations From an Idealised Reproduction

Being upfront about these, in the same spirit as the paper's disclosure policy:

- **`ml_cohort_tournament_v34.py` must be the cleaned version** (Subject 5 injection
  removed — see §2.1). If you run an un-cleaned copy, expect 75.27%/50.06% instead
  of 74.72%/47.88%. Both are documented in the paper (Section 10); only the
  cleaned numbers are the ones reported as final.
- **`ml_cohort_tournament_v34_1.py` should not be used at all** — confirmed
  duplicate of v34 (§2.4). Kept in this repo's history for transparency only.
- **Session timestamps in Stage 3 are generated at runtime**, not fixed — to
  reproduce the exact session 1784410860 numbers, use the logged CSVs in `data/`
  rather than regenerating a new session, which will produce a new (statistically
  similar, but not identical) timestamp and dataset.
- **The Mann–Whitney p-values reported in both the real-time engine (§6.3) and
  the STEW ERD test (§7.2) are intentionally not treated as significance claims**
  due to sample autocorrelation and pooling effects respectively — see the
  paper's explicit caveats in both sections before citing either p-value on
  its own.

# Leakage-Audited Riemannian Classification and Guardrail-Constrained AI Narration for Low-Channel EEG

**A verification-first BCI software pipeline, built from scratch during a gap year.**

🔁 **Reproduction Guide:** [`REPRODUCTION_GUIDE.md`](./REPRODUCTION_GUIDE.md) — exact terminal commands for every number in the paper
🗂️ **Script Index:** [`SCRIPT_INDEX.md`](./SCRIPT_INDEX.md) — which scripts are cited in the paper and which are exploratory context, so you don't have to guess
🐛 **Bug Registry:** [Appendix B of the paper](./paper/Paper5_FINAL.md#appendix-b-software-bug-registry) — every bug found, and how it was caught

---

## The Story

I started with just an idea: I wanted to build something that could track the growth of the brain — neuroplasticity — as a signal you could actually see change over time. I had just finished 12th boards. I had no background in neuroscience, no formal programming training, and no lab. What I had was a gap year, a terminal I didn't know how to use yet, and a stubborn habit — I like to fully imagine how a thing should work before I ever try to build it, the way Tesla is said to have worked. So before writing a line of code, I spent a long time going back and forth between Claude and Gemini, pasting one model's answer into the other, sharpening a rough idea into something with an actual roadmap.

That roadmap had three parts: a **hardware** front-end, an **application** layer for the user, and a **signal processing / ML** core. The hardware design exists — fully specified, verified in simulation — but I can't build the physical device yet; I'm a student with no funding, and that part is deliberately deferred to a future paper once I'm in college and can access a lab. The application layer is generic enough that it isn't a research contribution on its own. What's left — and what this repository is entirely about — is the signal processing and machine learning core, which turned out to be the part I could actually learn the most from, and the part rigorous enough to be worth publishing.

This repo is the record of four stages of that core, each one built on what the last stage taught me:

**Stage 1 — Can the pipeline even see through noise?** I started with the simplest possible fake brain: clean sine waves. That didn't last long — sine waves don't lie the way real EEG does, and a pipeline that only works on sine waves is a pipeline that's lying to you about how good it is. I replaced it with a stochastic simulator (noise-driven AR processes) and built the actual denoising machinery — FastICA and SOBI blind source separation — and benchmarked them against each other so future developers don't have to guess which one to reach for. This is the `BCI FINAL FILES` era: `sobi.py`, `riemannian.py`, `mini_eegnet.py`, `phase1_diagnostic.py`, `phase4_tournament.py`, `four_way_tournament.py`, `bootstrap_and_permutation.py`, `heuristic_sensitivity.py`.

**Stage 2 — Does it work on a real human brain, not just my simulation?** A pipeline that only works on data I made up isn't validated — it's just internally consistent. So I moved to PhysioNet's BCI2000 dataset, 50 real human subjects, and rebuilt the pipeline around it. This is where I spent the most time and made the most mistakes: **35 script revisions** (`ml_cohort_tournament_v10.py` through `v34`), each one catching a bug the last one didn't. The worst of them — a tangent-space alignment step that leaked information about the evaluation fold and produced 76% "accuracy" on pure random noise — is documented in the paper as a case study, because I think other people building BCI pipelines will hit the same trap if nobody warns them about it.

**Stage 3 — Wait, this dataset doesn't match my actual hardware.** Partway through Stage 2, I realized something I should have caught earlier: BCI2000 was recorded with a 64-channel wet-gel research cap. My target device is a 4-channel dry-electrode consumer form factor. Those are not the same signal regime, and no amount of channel-subsetting fixes that. I looked for a public 4-channel dry-electrode dataset — there isn't one. So instead of publishing a validation that quietly overstated what it proved, I built my own closed-loop 4-channel simulator, using everything Stage 1 and Stage 2 had taught me about what real EEG actually looks like. This stage also forced me to solve a problem I hadn't thought about at all going in: **every brain is a different size, and a BCI can't ship with one baseline value for everyone.** The calibration workflow below is how I solved that — think of it like a coach running tryouts before deciding how to coach each player, except the "player" is a simulated brain and the "tryout" is a 15-minute calibration session.

**Stage 4 — One more check, against a second real dataset.** Even with the closed-loop simulator working, I wanted an independent real-human check that wasn't the same dataset I'd already spent a month debugging against. That's the STEW dataset — 48 subjects, a completely different recording setup — used here purely as a geometric and spectral ground truth, not as a classification benchmark to chase.

I won't pretend I came out of this with expert-level command of neuroscience, biophysics, signal processing, hardware, and machine learning — that would be a strange thing for someone who finished school a year ago to claim, and it isn't true. What I can say is that I tried to be honest about every mistake I found along the way rather than quietly fixing and hiding them, and that discipline — verify, don't assume — is the actual contribution of this repository, more than any single accuracy number in it.

— Kaushik Pratap Singh

---

## Repository Structure

```
.
├── paper/
│   └── Paper5_FINAL.md              # The manuscript
├── figures/                          # All 8 paper figures, source-traceable to scripts below
├── REPRODUCTION_GUIDE.md             # Step-by-step commands to regenerate every number
├── stage1_signal_processing/         # BSS, filtering, sanity checks (see below)
├── stage2_bci2000_validation/        # ml_cohort_tournament_v10.py .. v34.py + logs
├── stage3_closed_loop_simulation/    # 4-channel simulator, calibration, real-time engine
├── stage4_stew_crossvalidation/      # Independent real-human cross-check
├── verification_scripts/             # loso_significance_test.py, proper_stats.py, etc.
└── data/                             # eeg_signals_*.csv, calibration_*.csv, structural_brain_baseline.npz
```

---

## Stage 1 — Signal Processing Foundations

*Goal: build and stress-test the denoising layer before trusting it on anything real.*

| Script | What it does |
|---|---|
| `sobi.py` | Second-Order Blind Identification — separates EEG sources via joint diagonalisation of time-lagged covariance matrices. Includes the known-answer test (two pure sinusoids, expected correlation 1.0000) that caught a Jacobi-rotation sign bug (Bug #5, see registry below). |
| `riemannian.py` | Core Riemannian geometry library: SPD matrix log/exp maps, Fréchet mean solver, geodesic distance. Includes a self-contained sanity check (feed it identical matrices, confirm the output error is at float64 machine-precision level). |
| `mini_eegnet.py` | A from-scratch convolutional network (no framework autodiff) with its own analytical backward pass, verified against a finite-difference numerical gradient check. |
| `phase1_diagnostic.py` / `phase_i_diagnostic.py` | Diagnoses *why* a BSS gate fails on real PhysioNet data — specifically, the sensitivity of SNR/correlation gates to which reference signal you compare against (unfiltered vs. matched-filter). This is the source of Figure 8 in the paper. |
| `phase2_sqi.py` | Signal Quality Index computation and lead-off (electrode disconnection) simulation. |
| `phase3_streaming_ola.py` | Real-time sliding-window Overlap-Add streaming implementation of the BSS pipeline, with a `RingBuffer` and fixed-calibration unmixing matrix. |
| `phase4_tournament.py` / `four_way_tournament.py` | The four-way classical/Riemannian/deep-learning regression tournament (Figure 6 in the paper). |
| `phase4_noise_sweep.py` | Extended noise-robustness sweep — checks whether the from-scratch CNN's clean-accuracy edge over classical ML survives when noise conditions shift away from the training distribution. |
| `ar4_benchmark.py` | A harder synthetic benchmark using AR(4)-filtered stochastic sources instead of pure sine tones, addressing the critique that pure sinusoids are an unrealistically easy signal model. |
| `rms_scaled_injection.py` | Scales injected blink/EMG artifacts to each channel's own baseline RMS, so injected noise amplitude is biophysically proportionate rather than a fixed constant across channels. |
| `bootstrap_and_permutation.py` | Adds bootstrap confidence intervals and permutation/paired significance testing on top of *already-computed* results (the FastICA/SOBI comparison and the ML tournament) — statistical context, not new claims. |
| `heuristic_sensitivity.py` | Sensitivity analysis of the pipeline's heuristic thresholds (e.g. SQI cutoff, gate thresholds). |
| `proper_stats.py` | Paired statistical testing specific to the Section 4.4 FastICA-vs-SOBI synthetic benchmark. *(Note: this script does **not** operate on LOSO fold accuracies — see `loso_significance_test.py` below for that.)* |

---

## Stage 2 — BCI2000 Validation (the 35-version gauntlet)

*Goal: prove the pipeline works on real human EEG, not just simulation. This is where almost every bug in Appendix B was found.*

- `phase1_physionet_validation.py` — downloads and validates PhysioNet BCI2000 recordings for use as the LOSO cohort.
- `batch_physionet_validation.py` — batch-processes the full 50-subject cohort.
- `ml_cohort_tournament_v10.py` → `ml_cohort_tournament_v34.py` — the version lineage. **Only `v34` (cleaned) is canonical; see the note below.**
- `subject_normalization_pipeline.py` — subject-specific baseline normalisation (annotation-aware active-epoch extraction with a diagnostic spectral probe).

**Key milestones in the version history** (full detail in the paper's Appendix B and Section 5.2):

| Version | What changed |
|---|---|
| v10 | First working LOSO tournament — but the TSA projector was fitted on REST-only trials, leaking 76.20% "accuracy" on pure Gaussian noise. |
| v11 | Fixed: TSA projector fitted globally across both classes. Noise accuracy collapses to chance (51.4%). |
| v29 → v30 | Found and fixed a silent subject-exclusion bug in the mean-accuracy accumulator. |
| v34 | Final canonical version. Originally shipped with a hardcoded noise injection affecting Subject 5 only (found during paper verification, removed — see Bug #7). |
| ~~v34.1~~ | **Do not use.** Confirmed via full-file diff and an independent rerun to be functionally identical to v34 once both were fixed of Bug #7 — a duplicate fork, not a distinct version. Kept in this repo's history for transparency, superseded by v34. |

**A note on why there are 35 versions of one script:** this isn't scope creep — it's the actual verification process. Every version bump in this lineage corresponds to either a bug found and fixed, or a check added to make sure a previous fix actually held. If you're building something similar, I'd genuinely recommend keeping every intermediate version the way I did — the diff between v10 and v11 is the single most useful artifact in this whole repository for understanding *how* the leakage bug worked, not just that it existed.

---

## Stage 3 — 4-Channel Closed-Loop Simulation

*Goal: since no public 4-channel dry-electrode dataset exists, build one — using everything Stage 1 and Stage 2 taught about what real EEG looks like — and solve per-user calibration properly.*

| Script | What it does |
|---|---|
| `virtual_brain_v4_lsl.py` | The 4-channel closed-loop AR(2) brain simulator, streamed over Lab Streaming Layer (LSL). Supports `--manual` mode for controlled calibration (see workflow below) and autonomous mode for realistic session evaluation. |
| `calibration_orchestrator_v2.py` | Runs the 3-phase, 15-minute calibration protocol against the manually-controlled simulator. |
| `realtime_inference_engine_v2_lsl.py` | The live closed-loop inference engine: reads calibration baselines, tracks the Riemannian Engagement Index in real time. |
| `evaluate_session_v2.py` | Post-session evaluation: reads the logged session CSVs, computes all Section 6.3 metrics (SQI, Cohen's d, geodesic distances, spectral checks), and produces the Figure 7 dashboard. |
| `lsl_real_time_receiver.py` / `live_receiver_pipeline.py` | LSL stream receivers used during live sessions. |
| `live_riemannian_scoring.py` / `live_bci_dashboard.py` / `live_tournament_pipeline.py` | Supporting live-session scoring and display utilities. |
| `wearable_inference_engine.py` / `bci_adaptive_stream_pipeline.py` / `bci_real_time_pipeline.py` | Earlier/alternate real-time pipeline implementations from this stage's development. |
| `phase5_calibration_corrected.py` | A corrected version of the per-user baseline calibration protocol (fixes a hardcoded-constant bug in the original data-quality gate calculation). |
| `mock_esp32_simulator.py` / `verify_stream.py` | Hardware-adjacent simulation and stream-integrity verification utilities, used to sanity-check the LSL pipeline ahead of eventual physical hardware integration. |

### The calibration workflow, step by step

This is the part that took the most trial and error to get right, so it's worth walking through explicitly — this is genuinely how you'd run it yourself:

1. **Open two terminals.** In Terminal 1, start the simulator in manual mode:
   ```
   py virtual_brain_v4_lsl.py --manual
   ```
2. **In Terminal 2, start the calibration orchestrator:**
   ```
   py calibration_orchestrator_v2.py
   ```
   It will prompt you to press Enter to begin the REST calibration phase.
3. **Back in Terminal 1**, type `1` and press Enter — this puts the simulated brain into the REST state.
4. **In Terminal 2**, press Enter to start the 4-minute REST calibration window. At the end of the 4 minutes, a baseline value is computed from that window and stored.
5. **Repeat for Cognition (`2` in Terminal 1) and Motor (`3` in Terminal 1)** — each gets its own 4-minute window and its own baseline.
6. Once all three phases are done, baselines are saved to `structural_brain_baseline.npz`, and the raw per-phase recordings are saved as `calibration_resting_alpha`, `calibration_cognitive_load`, and `calibration_motor_imagery`.
7. **Stop the manual simulator** in Terminal 1 (Ctrl+C).
8. **Restart the simulator in autonomous mode** (Terminal 1): `py virtual_brain_v4_lsl.py` — it now transitions between states on its own, the way a real, unscripted brain would.
9. **Start the inference engine** (Terminal 2): `py realtime_inference_engine_v2_lsl.py` — it loads the calibration baselines automatically and begins real-time tracking, saving `eeg_signals_<session>.csv` and `eeg_metrics_<session>.csv`.
10. **After a few minutes**, stop the inference engine (Ctrl+C) and, in the same terminal, run `py evaluate_session_v2.py` to generate the session evaluation report and dashboard figure.

The reasoning behind this: you can't ship one baseline value for every user — brains and skulls vary enough that a global calibration produces up to a 9.64 SD error for the most divergent subject (Section 5.4 of the paper). This workflow is the simulated equivalent of a coach running a short tryout with each new player before deciding how to coach them, done here in software because there's no physical subject yet to run it on.

---

## Stage 4 — STEW Cross-Validation

*Goal: an independent real-human check against a dataset that isn't the one this pipeline was debugged against for a month.*

| Script | What it does |
|---|---|
| `process_stew_benchmark.py` | Loads and preprocesses the 48-subject STEW dataset; computes per-subject 1/f spectral slopes and alpha-band metrics (Section 7.2). |
| `run_route_a_evaluation.py` | Riemannian-vs-Euclidean evaluation engine: computes the Fréchet mean, measures determinant-based matrix swelling on real human covariance matrices, and runs the window-count sensitivity sweep (Section 7.3). |
| `swelling_reproducibility_sweep` *(logged in `Swelling_Reproducibility_Sweep.txt`)* | The 20-seed reproducibility check that retracted an earlier, non-reproducible claim about a swelling non-monotonicity — see Section 9's discussion of self-correction. |

---

## AI Feedback Layer (Section 8 of the paper)

| Script | What it does |
|---|---|
| `llm_context_generator.py` | Builds the minimal fixed JSON schema handed from the signal-processing side (Stage A) to the LLM narration layer (Stage B) — session ID, engagement trend, data quality, nothing raw. |
| `app_backend.py` | Backend serving logic tying the inference engine, calibration, and AI narration layer together for the (generic, non-research) application layer. |

The guardrail itself (logit-bias blocklist + grammar-constrained decoding) is described in full in Section 8 of the paper; its 20-prompt structural coverage test is Table in Section 8.4.

---

## Verification & Reproducibility Scripts

| Script | What it does |
|---|---|
| `loso_significance_test.py` | **General-purpose, version-agnostic** paired Wilcoxon significance test for any `ml_cohort_tournament_(version).py` run — parses the standard fold-accuracy log format, pairs intact-label vs. shuffled-label runs by subject ID, reports W-statistic, p-value, bootstrap CI, and sign test. This replaced `proper_stats.py` for LOSO-level significance testing, since that script is specific to the Section 4.4 synthetic benchmark. Usage: `py loso_significance_test.py --intact <log.txt> --shuffled <log.txt> --metric TSA`. |
| `bci_recorder.py` / `bci_recorder_v2.py` | Session recording utilities used across development. |
| `bci_analysis.py` / `bci_analysis_enhancement.py` | Offline analysis utilities for recorded sessions. |
| `diagnostic_probe.py` | General-purpose diagnostic probing utility used during Stage 2/3 debugging. |
| `ml_data_processing.py` | Shared data-loading/preprocessing utilities used across the ML cohort scripts. |
| `online_streaming_pipeline.py` | Earlier online streaming pipeline prototype (superseded by Stage 3's real-time engine). |

---

## Figures

All figures are regenerated from the scripts above, not hand-edited. Source mapping:

| Figure | Source script(s) |
|---|---|
| Fig. 1 — Preprocessing chain diagram | Architecture summary (Section 4) |
| Fig. 2 — FastICA pipeline breakdown | Stage 1 BSS benchmark |
| Fig. 3 — FastICA vs. SOBI comparison | Stage 1 BSS benchmark |
| Fig. 4 — TSA leakage t-SNE (v10 vs. v11) | `ml_cohort_tournament_v10.py`, `ml_cohort_tournament_v11.py` |
| Fig. 5 — Final cohort t-SNE (n=50) | `ml_cohort_tournament_v34.py` (cleaned) |
| Fig. 6 — Four-way regression tournament | `phase4_tournament.py`, `four_way_tournament.py` |
| Fig. 7 — Session evaluation dashboard | `evaluate_session_v2.py` |
| Fig. 8 — PhysioNet Phase I diagnostic | `phase1_diagnostic.py` |

---

## Data Files

| File | Produced by | Contents |
|---|---|---|
| `calibration_resting_alpha.csv`, `calibration_cognitive_load.csv`, `calibration_motor_imagery.csv` | `calibration_orchestrator_v2.py` | Raw per-phase calibration recordings |
| `structural_brain_baseline.npz` | `calibration_orchestrator_v2.py` | Computed per-user baseline values |
| `eeg_signals_<session>.csv`, `eeg_metrics_<session>.csv` | `realtime_inference_engine_v2_lsl.py` | Raw signal and derived metrics for a live session |
| `swelling_reproducibility_results.csv` | `run_route_a_evaluation.py` (20-seed sweep) | Per-seed swelling ratios used in Section 7.3's reproducibility check |

---

## Known Bugs (full detail in the paper's Appendix B)

Every bug below was found through adversarial verification — a negative control, a known-answer test, or a sanity check specifically designed to catch it — not through code review alone. That's the actual point of this repository: **the verification process is a contribution in its own right, documented here with the same honesty as the paper.**

1. TSA alignment fitted on REST-only trials (v10) — leaked 76.20% accuracy on pure noise. Fixed in v11.
2. Silent subject exclusion from the mean-accuracy accumulator (v29). Fixed in v30.
3. Euclidean exponential-moving-average mislabelled as a Riemannian update — caused matrix swelling. Fixed.
4. Trace distance mislabelled as geodesic distance (2.6× error). Fixed.
5. SOBI Jacobi-rotation quadrant sign mismatch, caught by a known-answer test. Fixed.
6. MiniEEGNet gradient-check test harness index transposition. Fixed.
7. Hardcoded synthetic-noise injection affecting Subject 5 only, in `ml_cohort_tournament_v34.py`/`v34.1.py` — undisclosed in earlier drafts, found during paper verification, removed, full cohort re-run. See paper Section 10.
8. `v34.1` confirmed to be a duplicate fork of `v34`, not a distinct version — dropped.

---

## Citation

If you use this pipeline or its verification methodology, please cite:

```
Singh, K. P. (2026). Leakage-Audited Riemannian Classification and
Guardrail-Constrained AI Narration for Low-Channel EEG: A Verification
Methodology. 
```

---

## A closing note

I built this alone, during a gap year, with no lab and no prior background in most of the fields it touches. I'm not going to pretend that makes the results more impressive than they are — they're what they are, documented as honestly as I could manage. What I hope this repository is actually useful for, beyond the paper itself, is as a worked example: if you're a self-taught researcher hitting the same walls I hit — the leakage bug, the dataset-hardware mismatch, the calibration problem — the fix is in here, with the mistake that led to it left visible rather than cleaned up out of the final commit.

— K

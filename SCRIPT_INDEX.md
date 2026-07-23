# Script Index — What's Load-Bearing vs. What's Context

If you're trying to reproduce a specific claim in `paper/Paper5_FINAL.md`,
**start with `REPRODUCTION_GUIDE.md`, not this file.** The guide gives exact
commands, expected output, and runtime for every claim, in run order. This
index exists to answer a narrower question: *"why is this script in the
repo, and do I need to run it?"*

Every script in the repository falls into exactly one of three tiers below.

---

## ✅ REPRODUCTION_GUIDE.md corrected (resolved, not just flagged)

The guide's §1.2 has been edited directly: the command now reads
`py heuristic_sensitivity.py` instead of the incorrect `py phase1_diagnostic.py`.
`phase1_diagnostic.py` and `phase_i_diagnostic.py` are also relocated in the
guide's own repo-structure tree (§6) to Stage 2, matching this repository's
actual layout. Two documentation gaps were closed at the same time: §1.9 and
§1.10 were added to give `proper_stats.py` and `phase2_sqi.py` — both
referenced elsewhere in the guide but never given runnable commands — their
own reproduction steps. See the changelog at the top of `REPRODUCTION_GUIDE.md`
for the full list of edits made in this pass.

---

## Tier 1 — Load-bearing (cited in the paper, required to reproduce a claim)

These are the only scripts you need to run to reproduce every numbered claim
in the paper. Run order follows the paper's own section order (Stage 1 → 4).
Full commands and expected output for each are in `REPRODUCTION_GUIDE.md`
§1–§5 — **except §1.2, corrected above** — the table below is a map, not a
substitute for that guide.

| Folder | Script | Paper section | What it proves |
|---|---|---|---|
| `stage1_signal_processing/` | `sobi.py` | §4.1 | SOBI known-answer sanity check |
| `stage1_signal_processing/` | `heuristic_sensitivity.py` | §4.2–4.3 | **Corrected**: synthetic FastICA/SOBI SNR gain and correlation (+19.99 dB, 51.6%/43.2%) — not `phase1_diagnostic.py`, see error note above |
| `stage1_signal_processing/` | `bootstrap_and_permutation.py` | §4.4 | Paired FastICA vs. SOBI 60-trial significance test |
| `stage1_signal_processing/` | `proper_stats.py` | §4.4 | Canonical script for the synthetic FastICA-vs-SOBI benchmark per the paper's own Appendix note — distinct from `loso_significance_test.py`, which only supersedes it for the separate LOSO significance claim |
| `stage1_signal_processing/` | `phase2_sqi.py` | §4.3 | SQI/lead-off detachment validation (drops below 0.20 within 100ms) — the paper names the result, not the script by filename |
| `stage1_signal_processing/` | `ar4_benchmark.py` | §4.5 | AR(4) harder synthetic benchmark |
| `stage1_signal_processing/` | `mini_eegnet.py` | §5.1 | Gradient sanity check |
| `stage1_signal_processing/` | `riemannian.py` | §5.2 | Riemannian geometry sanity check |
| `stage1_signal_processing/` | `phase3_streaming_ola.py` | §5.4 | OLA streaming timing benchmark |
| `stage1_signal_processing/` | `phase4_tournament.py`, `four_way_tournament.py` | Fig. 6 | Four-way regression tournament |
| `stage1_signal_processing/` | `phase4_noise_sweep.py` | §5 (noise-robustness comparison vs. Paredes Ocaranza et al. 2025 [12]) | Extends the clean-accuracy tournament with a noise-severity sweep |
| `stage1_signal_processing/` | `rms_scaled_injection.py` | Referenced in `phase1_diagnostic.py`'s docstring as the fix for fixed-amplitude artifact injection | RMS-scaled blink/EMG artifact injection, channel-relative |
| `stage2_bci2000_validation/` | `ml_cohort_tournament_v10.py` | Fig. 4 (left) | TSA leakage bug demonstration (superseded — kept as the case study) |
| `stage2_bci2000_validation/` | `ml_cohort_tournament_v11.py` | Fig. 4 (right) | TSA leakage fix confirmation |
| `stage2_bci2000_validation/` | `ml_cohort_tournament_v34.py` | Fig. 5, §5.3 | **Final canonical cohort result** — 74.72% intact vs. 47.88% shuffled |
| `stage2_bci2000_validation/` | `phase1_physionet_validation.py`, `phase1_diagnostic.py`, `phase_i_diagnostic.py` | Fig. 8, §10 | **Moved here from Stage 1** — real-data SNR/correlation diagnostic, matched-reference branch (−0.34 dB / 41.4%) |
| `stage2_bci2000_validation/` | `subject_normalization_pipeline.py` | §6.2 | 13.1× EI variability across subjects |
| `stage3_closed_loop_simulation/` | `virtual_brain_v4_lsl.py`, `calibration_orchestrator_v2.py`, `realtime_inference_engine_v2_lsl.py`, `evaluate_session_v2.py` | Fig. 7, §6.3 | Session 1784410860 — SQI, Cohen's d, geodesics, ERD/ERS |
| `stage4_stew_crossvalidation/` | `process_stew_benchmark.py` | §9 | STEW 1/f slope and alpha ERD null result |
| `stage4_stew_crossvalidation/` | `run_route_a_evaluation.py` | Table II | Swelling range, Riemannian volume preservation |
| `verification_scripts/` | `loso_significance_test.py` | §5.3 | Wilcoxon W=66.0, p=1.36×10⁻⁷ significance test |
| `ai_narration_layer/` | `llm_context_generator.py` | §8.4 | CFG grammar the 20-prompt guardrail coverage argument is verified against (inspection, not execution — see note below) |

**Resolved (author-confirmed):** `swelling_reproducibility_sweep.py` is the
canonical script for the 20-seed swelling reproducibility sweep and its
retraction (paper §9, finding #3) — it belongs in `stage4/` only.
`heuristic_sensitivity.py` is the canonical script for the synthetic
FastICA/SOBI benchmark (§4.2–4.3) — it belongs in `stage1/` only. The
duplicate copy of `heuristic_sensitivity.py` previously placed in `stage4/`
has been removed to eliminate the ambiguity flagged in an earlier version of
this file and in `REPRODUCTION_GUIDE.md` §4.3.

---

## Tier 2 — Exploratory / historical, not required for any paper claim

Located under `exploratory_not_required_for_paper/`. Nothing here is cited
by section number in the paper. Kept for transparency of the debugging
journey (consistent with this project's explicit-disclosure principle), not
because a reader needs to run any of it to reproduce a result.

- **`ml_cohort_lineage_v12_to_v33/`** — 22 intermediate versions of the ML
  cohort tournament script between the leakage-bug version (v11) and the
  final cleaned version (v34). `REPRODUCTION_GUIDE.md` §2.1 explicitly says:
  *"Use only `ml_cohort_tournament_v34.py`, cleaned"* — v10 and v11 are kept
  for the leakage case study, and v29→v30 is mentioned narratively (a silent
  subject-exclusion bug found and fixed), but v12–v28 and v31–v33 are not
  individually referenced anywhere in the paper or the guide. They're the
  version-control history of the debugging process, included for anyone who
  wants to trace how the pipeline evolved, not because any of them backs a
  specific number in the manuscript.
- **`live_deployment_stack/`** — a parallel set of scripts for a real-time,
  wearable/streaming deployment (dashboard, recorder, ESP32 simulator,
  adaptive streaming pipeline, live tournament/riemannian-scoring variants,
  a Flask-style `app_backend.py`, etc.). None of these are cited in
  `Paper5_FINAL.md`. They represent exploratory work toward a future
  hardware-integrated system, outside the scope of this Methods-article
  submission, which is explicitly a software-verification study with no
  device or human-subject data (see the paper's Author Note).

If you're only trying to reproduce the paper, you can ignore this entire
folder.

---

## Tier 3 — Documentation, data, and figures

Not scripts — see `README.md` for the data file table and `figures/README.md`
for figure provenance (including two disclosed issues there worth reading
before citing those figures).

---

## Quick answer to "what do I run and in what order?"

1. `stage1_signal_processing/` scripts (any order, all independent, <5 min each)
2. `stage2_bci2000_validation/`: v10 → v11 → v34 → `loso_significance_test.py` → `subject_normalization_pipeline.py`
3. `stage3_closed_loop_simulation/`: `virtual_brain_v4_lsl.py` (Terminal 1) → `calibration_orchestrator_v2.py` (Terminal 2) → `realtime_inference_engine_v2_lsl.py` (Terminal 3) → `evaluate_session_v2.py`
4. `stage4_stew_crossvalidation/`: `process_stew_benchmark.py` → `run_route_a_evaluation.py` → `swelling_reproducibility_sweep.py`

For exact commands, flags, and expected numeric output at every step, use
`REPRODUCTION_GUIDE.md` — this file only tells you which folder to look in
and why each script exists.

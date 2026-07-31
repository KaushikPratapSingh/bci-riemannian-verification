# Section 10 Open-Problem Investigation

This folder contains the complete, reproducible trail behind the paper's
disclosed open problem (Section 10; Appendix B items 9-11): the discovery
that the paper's originally-stated causes for the live-session 1/f slope
steepening and the beta ERD/ERS direction anomaly were both wrong, the
correct mechanisms and their fixes, and the still-unresolved trade-off
between noise-floor tuning and Riemannian state-separability.

**Start here:** [`../OPEN_PROBLEM_noise_floor_tradeoff.md`](../OPEN_PROBLEM_noise_floor_tradeoff.md)
at the repo root is the full narrative writeup. This folder is the
supporting evidence and code referenced throughout it.

## Two resolved fixes (verified, not open)

| Finding | Script(s) | Patch |
|---|---|---|
| 1/f slope steepening misdiagnosed as causal-filter group delay; real cause was the slope-fit window (4-40 Hz) extending into the filter's own 45 Hz corner | `groupdelay_experiment.py`, `diagnose_slope.py`, `fix2_scan_fmax.py`, `fix2_final_verify.py` | `patches/fix2_evaluate_session_v2.patch.md` |
| Beta ERD/ERS direction anomaly misdiagnosed as AR(2) cross-frequency coupling; real cause was alpha oscillator damping (r=0.93) letting ~19% of its own power bleed into the beta band | `beta_anomaly_check.py`, `fix3_beta_leakage.py`, `fix3_oscillator_test.py`, `fix3_full_pipeline_test.py` | `patches/fix3_virtual_brain_v4_lsl.patch.md` |

Both fixes are narrow, mechanistically verified, and considered closed.

## The open problem: noise-floor vs. separability trade-off

| Investigation step | Script(s) | Patch |
|---|---|---|
| Ruled out: A_MIX spatial-mixing-weight rebalancing as the fix for residual channel-specific slope steepness | `fix2b_amix_beta_weight.py`, `fix2b_amix_beta_weight_v2.py`, `fix2c_theta_weight_test.py` | `patches/fix2b_noise_floor_amix.patch.md` |
| Identified: flat broadband noise-floor amplitude as the real (but double-edged) lever | `fix2d_noise_floor_test.py`, `fix2e_side_effects_check.py` | same |
| Tested at the actual live pink_std=8 level (not just the offline 2.5 baseline) | `fix2f_fmax_rescan_at_p8.py` | — |
| State-dependent (REST/COGNITIVE/MOTOR) noise floor, literature-grounded, offline-verified | `fix_state_scaled_noise.py`, `fix_state_scaled_v3.py` | `patches/fix_final_state_scaled_noise.patch.md` |
| Critical confound discovered: `virtual_brain_v4_lsl.py`'s subject-profile and noise-generator random streams were seeded from wall-clock time, confounding every live parameter comparison | — | `patches/CRITICAL_fix_confounded_rng.patch.md` |
| Independent recomputation confirming a specific run's numbers were computed correctly (ruling out a bug in the npz generation itself, as distinct from the seed confound) | `verify_p11_cogmotor.py` | — |

## `live_run_evidence/` — real data, not just narration

Every live pipeline run referenced in the open-problem writeup, in order:

| pink_std / fmax value | Files | Outcome |
|---|---|---|
| 6 | `structural_brain_baseline_p_6.npz`, `bci_evaluation_v2_p_6.png` | Slope fails (too steep), all distances pass |
| 8 | `structural_brain_baseline_p_8.npz`, `bci_evaluation_v2_p_8.png` | Slope close but fails, all distances pass |
| 10 | `structural_brain_baseline_p_10.npz`, `bci_evaluation_v2_p_10.png` | Slope fails (worse than 8 -- run-to-run variance before the RNG fix), all distances pass |
| 11 (run 1, pre-seed-fix) | `structural_brain_baseline_p_11.npz`, `bci_evaluation_v2_p_11.png`, `calibration_*_p_11.csv`, `eeg_signals_1784953702_p_11.csv` | Slope passes, Cog<->Motor distance fails |
| 30 | `structural_brain_baseline_p_30.npz`, `bci_evaluation_v2_p_30.png` | Slope passes, all three distances collapse badly |
| fmax=25 (pink_std unchanged) | `structural_brain_baseline_fmax_25.npz`, `bci_evaluation_v2_fmax_25.png` | Slope still fails; Rest<->Cog distance also fails here (a different failure than the Cog<->Motor one seen elsewhere) |
| fmax=26 (pink_std unchanged) | `structural_brain_baseline_fmax_26.npz`, `bci_evaluation_v2_fmax_26.png` | Slope still fails; distances pass |
| State-scaled, post-seed-fix (run 2) | `structural_brain_baseline_postseedfix_run2.npz`, `bci_evaluation_v2_postseedfix_run2.png`, `bci_evaluation_report_postseedfix_run2.txt`, `eeg_signals_1784958275.csv`, `eeg_metrics_1784958275.csv` | Slope fails, all three distances pass -- the reproducible post-seed-fix pattern documented in the open-problem writeup |

The `calibration_resting_alpha_p_11.csv` / `calibration_cognitive_load_p_11.csv` /
`calibration_motor_imagery_p_11.csv` triplet is included specifically because
`verify_p11_cogmotor.py` uses them to independently re-derive
`structural_brain_baseline_p_11.npz`'s Cog<->Motor distance from raw data,
confirming the number was computed correctly (i.e., the failure at pink_std=11
run 1 was real, not a computation bug).

## What's still missing

The exact REST_STD/COGNITIVE_STD/MOTOR_STD values used for the two
post-seed-fix confirmatory runs referenced in the open-problem writeup were
not logged at the time and could not be recovered before this investigation
paused (see the writeup's Section 8, item 1). Whoever picks this up next
should modify `virtual_brain_v4_lsl.py` to print its active noise-floor
configuration to the terminal banner at startup (the same way it already
prints the `SUBJECT` profile) so this doesn't happen again.

# Figures

All figures below are raw outputs copied directly from the scripts named in
the main README's Figure table. Filenames indicate the source script,
version, and condition (intact-label run vs. shuffled-label negative
control).

| File | Source | Condition |
|---|---|---|
| `Figure4a_TSA_leakage_tsne_v10_intact.png` | `ml_cohort_tournament_v10.py` | Intact labels |
| `Figure4b_TSA_leakage_tsne_v10_negative_control.png` | `ml_cohort_tournament_v10.py --run-negative-control` | Shuffled labels |
| `Figure4c_TSA_leakage_tsne_v11_intact.png` | `ml_cohort_tournament_v11.py` | Intact labels |
| `Figure4d_TSA_leakage_tsne_v11_negative_control.png` | `ml_cohort_tournament_v11.py --run-negative-control` | Shuffled labels |
| `Figure5a_final_cohort_tsne_v34_intact.png` | `ml_cohort_tournament_v34.py` | Intact labels (t-SNE) |
| `Figure5b_final_cohort_tsne_v34_negative_control.png` | `ml_cohort_tournament_v34.py --run-negative-control` | Shuffled labels (t-SNE) |
| `Figure5c_v34_performance_intact.png` | `ml_cohort_tournament_v34.py` | Intact labels (bar + fold-accuracy chart) |
| `Figure5d_v34_performance_negative_control.png` | `ml_cohort_tournament_v34.py --run-negative-control` | Shuffled labels (bar + fold-accuracy chart) |

## Resolved — false alarm on the v10/v11 "identical images" flag

An earlier version of this note flagged `Figure4a`/`Figure4b` (v10) and
`Figure4c`/`Figure4d` (v11) as byte-identical (same MD5 within each pair) and
suggested the negative-control run might not have actually happened.

That was wrong. `ml_cohort_tournament_v10.py` and `_v11.py` both branch their
save filename correctly on `--run-negative-control`, and the terminal
transcripts (`ML_Cohort_v10.txt`, `ML_Cohort_v11.txt`) confirm two genuinely
separate executions with two distinct results — e.g. for v10, the intact run's
Ridge (Optimized GCV) scored 70.14% vs. 44.82% for the shuffled-label run. The
negative control worked as intended on both versions.

The identical-hash images currently in this folder are almost certainly a
duplicate-upload artifact (the same file selected twice among eight similarly
named PNGs), not evidence of a data problem. **Action needed:** re-export or
re-upload the correct, distinct `ml_cohort_tournament_v10_negative_control.png`
and `ml_cohort_tournament_v11_negative_control.png` from the original run
output on your machine to replace the current duplicates.

The v34 figures (`Figure5a`–`Figure5d`) were already confirmed distinct by
hash and need no action.

## ⚠️ Second discrepancy — these are not yet the paper's assembled Figure 4

The paper (Section 5.2, lines ~158–162) describes **Figure 4** as a single
composite: a left/right pure-noise scatter comparison (v10 leaky vs. v11
fixed), plus a "continued" three-panel t-SNE (1. raw spatial covariance,
2. standard tangent space, 3. corrected TSA-aligned). What's in this folder
right now (`Figure4a`–`Figure4d`) are four **separate, per-version, per-condition**
diagnostic plots — each one already showing all three panels on its own —
not the single assembled figure the manuscript references by filename
(`figures/Figure4_leakage_tsne.png`).

Before submission, either:
1. Assemble the actual `Figure4_leakage_tsne.png` composite the caption
   describes (v10 intact left panel + v11 intact right panel, or whichever
   sub-panels the caption is actually pointing at), or
2. Update the caption/filename reference in the paper to match what these
   four raw per-version plots actually show, rather than leaving a filename
   in the manuscript that doesn't correspond to any single file in the repo.

Flagging this now rather than renaming these four files to `Figure4.png` and
letting the mismatch pass silently.

Per this repository's verification-first principle (see Appendix B / README
"Known Bugs"): **do not silently regenerate or relabel these v10/v11 figures
to make them look distinct.** If you still have the original separate
negative-control runs for v10/v11 on your machine, replace the duplicated
files with the real ones. If the negative-control run for those two early
versions was genuinely never executed, the honest fix is to say so explicitly
wherever Figure 4 is referenced in the paper, rather than implying two
independent verifications occurred.

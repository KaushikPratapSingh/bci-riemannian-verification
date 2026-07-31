"""
loso_significance_test.py

General-purpose, version-agnostic significance test for LOSO fold accuracies
produced by ANY ml_cohort_tournament_(version).py run (v10 through v34+).

Unlike proper_stats.py -- which is hard-wired to phase4_tournament.py's
synthetic-session simulator -- this script only depends on the plain-text
terminal log format that every ml_cohort_tournament version prints:

  S01 fold | RF Raw: 73.3% | TSA: 50.0% || SVM Raw: 70.0% | Gated SVM: 56.7% | ...

It extracts per-subject fold accuracies for a chosen metric column (default:
"TSA", i.e. RF on TSA features -- the paper's headline metric), pairs subjects
by ID across two runs (typically an intact-label run and a shuffled-label
negative-control run), and reports:

  - Wilcoxon signed-rank test (paired, non-parametric; appropriate here
    because n=50 subject-level accuracies are not assumed normally distributed
    and the two runs are paired by subject)
  - Paired mean difference and its 95% bootstrap CI
  - Sign test agreement (fraction of subjects where run A > run B)

USAGE
-----
    py loso_significance_test.py --intact ML_Cohort_v34.txt --shuffled ML_Cohort_v34.txt --metric TSA

If both runs are in the same combined log (as with `--run-negative-control`
appended output), pass the same file to both --intact and --shuffled; the
script automatically finds the two separate
"RUNNING NESTED LEAVE-ONE-SUBJECT-OUT" blocks and treats the first as intact
and the second as shuffled. To compare two separate log files instead, pass
each file path independently.

Supported --metric values: "RF Raw", "TSA", "SVM Raw", "Gated SVM"
(matches the column names printed in the fold line).
"""

import argparse
import re
import sys
import numpy as np
from scipy.stats import wilcoxon

FOLD_LINE_RE = re.compile(
    r"S(\d+)\s+fold\s*\|\s*RF Raw:\s*([\d.]+)%\s*\|\s*TSA:\s*([\d.]+)%\s*\|\|\s*"
    r"SVM Raw:\s*([\d.]+)%\s*\|\s*Gated SVM:\s*([\d.]+)%"
)

METRIC_COLUMN = {
    "RF Raw": 1,
    "TSA": 2,
    "SVM Raw": 3,
    "Gated SVM": 4,
}

BLOCK_START_MARKER = "RUNNING NESTED LEAVE-ONE-SUBJECT-OUT"


def extract_blocks(text):
    """Split a terminal log into one or more LOSO fold blocks, returning a
    list of {subject_id: [rf_raw, tsa, svm_raw, gated_svm]} dicts, one per
    block, in the order they appear."""
    block_starts = [m.start() for m in re.finditer(BLOCK_START_MARKER, text)]
    if not block_starts:
        raise ValueError(
            f"No '{BLOCK_START_MARKER}' block found in this log. "
            "Is this a raw ml_cohort_tournament_(version).py terminal transcript?"
        )
    block_starts.append(len(text))
    blocks = []
    for i in range(len(block_starts) - 1):
        chunk = text[block_starts[i]:block_starts[i + 1]]
        subjects = {}
        for m in FOLD_LINE_RE.finditer(chunk):
            sid = int(m.group(1))
            subjects[sid] = [float(m.group(j)) for j in (2, 3, 4, 5)]
        if subjects:
            blocks.append(subjects)
    if not blocks:
        raise ValueError("Found LOSO block marker(s) but no parseable fold lines beneath them.")
    return blocks


def paired_arrays(block_a, block_b, metric):
    col = METRIC_COLUMN[metric]
    common_ids = sorted(set(block_a) & set(block_b))
    if len(common_ids) < len(block_a) or len(common_ids) < len(block_b):
        missing_a = set(block_b) - set(block_a)
        missing_b = set(block_a) - set(block_b)
        if missing_a or missing_b:
            print(
                f"[warning] subject mismatch between runs -- "
                f"missing from A: {sorted(missing_a)}, missing from B: {sorted(missing_b)}. "
                f"Proceeding with {len(common_ids)} common subjects.",
                file=sys.stderr,
            )
    a = np.array([block_a[sid][col - 1] for sid in common_ids])
    b = np.array([block_b[sid][col - 1] for sid in common_ids])
    return common_ids, a, b


def bootstrap_ci_of_mean_diff(diffs, n_boot=10000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boots = np.array([diffs[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi


def run(intact_path, shuffled_path, metric):
    with open(intact_path, "r", encoding="utf-8", errors="replace") as f:
        intact_text = f.read()

    if shuffled_path == intact_path:
        blocks = extract_blocks(intact_text)
        if len(blocks) < 2:
            raise ValueError(
                f"--intact and --shuffled point to the same file, but only "
                f"{len(blocks)} LOSO block(s) were found in it. Provide separate "
                f"log files, or a combined log containing both the intact-label "
                f"run and the --run-negative-control run."
            )
        block_intact, block_shuffled = blocks[0], blocks[1]
    else:
        with open(shuffled_path, "r", encoding="utf-8", errors="replace") as f:
            shuffled_text = f.read()
        block_intact = extract_blocks(intact_text)[0]
        block_shuffled = extract_blocks(shuffled_text)[0]

    ids, a, b = paired_arrays(block_intact, block_shuffled, metric)
    n = len(ids)
    diffs = a - b

    print(f"Metric: {metric}  |  n subjects paired: {n}")
    print(f"Intact   mean: {a.mean():.2f}%  (min {a.min():.1f}, max {a.max():.1f})")
    print(f"Shuffled mean: {b.mean():.2f}%  (min {b.min():.1f}, max {b.max():.1f})")
    print(f"Mean paired difference (intact - shuffled): {diffs.mean():.2f} pp")

    ci_lo, ci_hi = bootstrap_ci_of_mean_diff(diffs)
    print(f"95% bootstrap CI on mean difference: [{ci_lo:.2f}, {ci_hi:.2f}] pp")

    n_ties = int(np.sum(diffs == 0))
    n_pos = int(np.sum(diffs > 0))
    n_neg = int(np.sum(diffs < 0))
    print(f"Sign test: intact > shuffled for {n_pos}/{n} subjects, "
          f"< for {n_neg}/{n}, tied for {n_ties}/{n}")

    if n_ties == n:
        print("\nAll paired differences are zero -- Wilcoxon signed-rank test is undefined.")
        return

    try:
        stat, p = wilcoxon(a, b, zero_method="wilcox")
        print(f"\nWilcoxon signed-rank test (paired, two-sided): "
              f"W={stat:.1f}, p={p:.6f} -> "
              f"{'significant' if p < 0.05 else 'NOT significant'} at alpha=0.05")
    except ValueError as e:
        print(f"\nWilcoxon test could not be computed: {e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--intact", required=True, help="Path to the intact-label run's terminal log (.txt)")
    parser.add_argument("--shuffled", required=True, help="Path to the shuffled-label (--run-negative-control) run's terminal log (.txt). Can be the same file as --intact if it contains both blocks.")
    parser.add_argument("--metric", default="TSA", choices=list(METRIC_COLUMN.keys()), help="Which fold-accuracy column to test (default: TSA)")
    args = parser.parse_args()
    run(args.intact, args.shuffled, args.metric)


if __name__ == "__main__":
    main()
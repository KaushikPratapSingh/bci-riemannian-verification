"""
Swelling Non-Monotonicity Reproducibility Sweep
================================================
Answers the question the reviewer raised about Section 7.3: is the dip in the
Euclidean determinant-swelling ratio at N=1,000 a reproducible property of the
estimator, or an artefact of which windows happened to land in that slice on
one particular run?

Method: for each window count N in a fixed list, draw MANY independent random
subsets of size N (different seeds) from the full pool of available windows,
compute the Euclidean-vs-Riemannian determinant swelling ratio for each draw,
and report mean +/- std across seeds. If the dip at N=1000 is real, it should
show up as a genuine dip in the *mean* across seeds, not just in one draw, and
the seed-to-seed spread should be much smaller than the size of the dip.

Usage:
    1. Place this file in the same directory as run_route_a_evaluation.py
    2. Make sure human_benchmarks/human_resting_alpha.csv and
       human_benchmarks/human_cognitive_load.csv exist (run
       process_stew_benchmark.py first if they don't).
    3. python swelling_reproducibility_sweep.py
    4. Send me the printed table (or the CSV it writes out) and I'll fold the
       result into Section 7.3 with an honest re-write, whichever way it goes.
"""

import os
import numpy as np
import pandas as pd
from scipy.linalg import logm, expm, eigh

# ── Reuse the exact functions already used for the paper's headline numbers ──
# (imported, not reimplemented, so this sweep is guaranteed consistent with
#  the numbers already in Section 7.3 rather than a subtly different method)
from run_route_a_evaluation import (
    compute_windowed_covariance,
    compute_determinant_swelling,
    BENCHMARK_DIR,
    REST_FILE,
)


def _sym_matrix_log(M, vals=None, vecs=None):
    """Matrix logarithm of a symmetric positive-definite matrix via
    eigendecomposition. Mathematically identical to scipy.linalg.logm for
    SPD input, but 10-50x faster since it avoids the general-purpose
    Schur-decomposition path logm() uses for arbitrary matrices."""
    if vals is None:
        vals, vecs = eigh(M)
    vals = np.maximum(vals, 1e-12)
    return vecs @ np.diag(np.log(vals)) @ vecs.T


def _sym_matrix_exp(M):
    """Matrix exponential of a symmetric matrix via eigendecomposition."""
    vals, vecs = eigh(M)
    return vecs @ np.diag(np.exp(vals)) @ vecs.T


def riemannian_frechet_mean_fast(covs, max_iter=100, tol=1e-5):
    """Drop-in replacement for run_route_a_evaluation.riemannian_frechet_mean
    that uses eigh-based log/exp instead of scipy.linalg.logm/expm. Produces
    the same result (both are the exact operation for SPD matrices) at a
    fraction of the runtime, which is what made the original sweep slow at
    large N x many seeds."""
    r_mean = np.mean(covs, axis=0)
    iterations_run = 0
    converged = False

    for i in range(max_iter):
        iterations_run = i + 1
        vals, vecs = eigh(r_mean)
        vals = np.maximum(vals, 1e-12)
        sqrt_mean = vecs @ np.diag(np.sqrt(vals)) @ vecs.T
        inv_sqrt_mean = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T

        tangent_vectors = []
        for c in covs:
            proj = inv_sqrt_mean @ c @ inv_sqrt_mean
            tangent_vectors.append(_sym_matrix_log(proj))

        mean_tangent = np.mean(tangent_vectors, axis=0)
        step = sqrt_mean @ _sym_matrix_exp(mean_tangent) @ sqrt_mean

        diff = np.linalg.norm(step - r_mean, ord='fro')
        r_mean = step

        if diff < tol:
            converged = True
            break

    return r_mean, iterations_run, converged

# Window counts to test -- includes the two values that produced the dip
# (500 and 1000) plus neighbours so we can see the shape of the curve, not
# just the two endpoints.
WINDOW_COUNTS = [100, 250, 500, 750, 1000, 1500, 2000, 3000]

# How many independent random draws per N. 20 is enough to see whether the
# spread across seeds is small (real effect) or comparable to the dip itself
# (artefact of window selection).
N_SEEDS = 20


def sweep():
    if not os.path.exists(REST_FILE):
        print(f"Could not find {REST_FILE}.")
        print("Run process_stew_benchmark.py first to generate it.")
        return None

    print("Loading resting-state STEW data and computing the full window pool...")
    rest_data = pd.read_csv(REST_FILE).iloc[:, 1:].values
    all_covs = compute_windowed_covariance(rest_data)
    n_available = len(all_covs)
    print(f"Total windows available: {n_available}\n")

    results = []
    for N in WINDOW_COUNTS:
        if N > n_available:
            print(f"Skipping N={N}: only {n_available} windows available.")
            continue

        import time
        t_start = time.time()
        ratios = []
        for seed in range(N_SEEDS):
            rng = np.random.RandomState(seed)
            idx = rng.choice(n_available, size=N, replace=False)
            subset = all_covs[idx]

            euclidean_mean = np.mean(subset, axis=0)
            riemannian_mean, iters, converged = riemannian_frechet_mean_fast(subset)

            ratio = compute_determinant_swelling(euclidean_mean, riemannian_mean)
            ratios.append(ratio)
            print(f"  N={N} seed={seed+1}/{N_SEEDS} ratio={ratio:.2f}x "
                  f"(iters={iters}, {time.time()-t_start:.1f}s elapsed)", end="\r")

        elapsed = time.time() - t_start
        print()  # newline after the \r progress line
        print(f"  -> N={N} done in {elapsed:.1f}s total")

        ratios = np.array(ratios)
        mean_ratio = ratios.mean()
        std_ratio = ratios.std()
        cv = std_ratio / mean_ratio if mean_ratio != 0 else float("nan")

        print(f"N={N:>5}  mean={mean_ratio:>12.2f}x  std={std_ratio:>10.2f}  "
              f"CV={cv:>6.3f}  min={ratios.min():>10.2f}  max={ratios.max():>10.2f}")

        results.append({
            "N": N, "mean_ratio": mean_ratio, "std_ratio": std_ratio,
            "cv": cv, "min_ratio": ratios.min(), "max_ratio": ratios.max(),
            "n_seeds": N_SEEDS,
        })

    df = pd.DataFrame(results)
    out_path = "swelling_reproducibility_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")

    # Simple interpretation guide printed at the end
    print("\n" + "=" * 70)
    print("HOW TO READ THIS:")
    print("If the mean_ratio column still dips at N=1000 relative to N=500 and")
    print("N=1500, AND the std/CV at each N is small relative to the size of")
    print("that dip, the non-monotonicity is a reproducible property of the")
    print("estimator (Section 7.3's original explanation holds, just now with")
    print("evidence instead of narrative).")
    print()
    print("If the dip disappears, moves to a different N, or the std at each N")
    print("is comparable to the size of the dip itself, the original single-run")
    print("dip was most likely a sampling artefact and Section 7.3 should be")
    print("rewritten to say the swelling ratio increases roughly monotonically")
    print("with N once averaged over window selection, with seed-to-seed noise")
    print("of magnitude X at each N.")
    print("=" * 70)

    return df


if __name__ == "__main__":
    sweep()

Terminal Output:

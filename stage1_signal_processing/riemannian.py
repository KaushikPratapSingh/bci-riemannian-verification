"""
Riemannian-geometry features for the ML tournament (review item IV.3).

Barachant, Bonnet, Congedo, Jutten (2012, IEEE TBME) represent each EEG
trial as its spatial covariance matrix and classify by distance in the
Riemannian manifold of symmetric positive-definite (SPD) matrices, or (their
second method, "tangent space LDA") by projecting covariance matrices onto
the tangent space at the geometric mean and classifying the resulting
Euclidean vectors with LDA.

This project's task is regression (predicting a continuous focus score),
not classification, so what's implemented here is the regression analogue
of their second method: project to the tangent space, then run Ridge
regression instead of LDA. This is an adaptation of their published
representation, not a reproduction of their exact (classification)
algorithm -- that distinction is kept explicit rather than implied away.
"""

import numpy as np


def sym_matrix_op(C, func):
    """Applies a scalar function to the eigenvalues of a symmetric matrix
    and reconstructs -- the standard way to define sqrt/log/exp of an SPD
    matrix, since these are only well-defined via the eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(C)
    return eigvecs @ np.diag(func(eigvals)) @ eigvecs.T


def sqrtm_spd(C):
    return sym_matrix_op(C, np.sqrt)


def invsqrtm_spd(C):
    return sym_matrix_op(C, lambda v: 1.0 / np.sqrt(np.clip(v, 1e-12, None)))


def logm_spd(C):
    return sym_matrix_op(C, lambda v: np.log(np.clip(v, 1e-12, None)))


def expm_sym(C):
    return sym_matrix_op(C, np.exp)


def riemannian_mean(covs, n_iter=50, tol=1e-8):
    """Karcher/Frechet mean of a set of SPD matrices via the standard
    gradient-descent algorithm on the affine-invariant Riemannian metric."""
    C_mean = np.mean(covs, axis=0)  # Euclidean mean as starting point
    for _ in range(n_iter):
        C_mean_sqrt = sqrtm_spd(C_mean)
        C_mean_invsqrt = invsqrtm_spd(C_mean)
        tangent_sum = np.zeros_like(C_mean)
        for C in covs:
            tangent_sum += logm_spd(C_mean_invsqrt @ C @ C_mean_invsqrt)
        tangent_mean = tangent_sum / len(covs)
        update = C_mean_sqrt @ expm_sym(tangent_mean) @ C_mean_sqrt
        if np.linalg.norm(update - C_mean) < tol:
            C_mean = update
            break
        C_mean = update
    return C_mean


def tangent_space_vector(C, C_ref_invsqrt):
    """Projects covariance matrix C onto the tangent space at the reference
    mean, then vectorizes the symmetric tangent matrix preserving the
    Frobenius inner product (off-diagonal entries scaled by sqrt(2))."""
    S = logm_spd(C_ref_invsqrt @ C @ C_ref_invsqrt)
    n = S.shape[0]
    vec = []
    for i in range(n):
        for j in range(i, n):
            vec.append(S[i, j] if i == j else S[i, j] * np.sqrt(2))
    return np.array(vec)


def covariance_features(window_4ch):
    """Spatial covariance matrix for one window, regularized slightly for
    numerical stability (standard practice -- an unregularized sample
    covariance from a short window can be ill-conditioned)."""
    Xc = window_4ch - window_4ch.mean(axis=0, keepdims=True)
    C = (Xc.T @ Xc) / Xc.shape[0]
    C += 1e-6 * np.trace(C) / C.shape[0] * np.eye(C.shape[0])
    return C


def sanity_check():
    """Known-answer check before trusting this on real data: the Riemannian
    mean of N copies of the SAME SPD matrix must equal that matrix exactly,
    and the tangent-space vector of a matrix at its own mean must be all
    zeros (since log of the identity-transported matrix is zero)."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((4, 4))
    C = A @ A.T + 4 * np.eye(4)  # a fixed, arbitrary SPD matrix

    covs = [C.copy() for _ in range(10)]
    C_mean = riemannian_mean(covs)
    err1 = np.max(np.abs(C_mean - C))

    C_ref_invsqrt = invsqrtm_spd(C_mean)
    vec = tangent_space_vector(C, C_ref_invsqrt)
    err2 = np.max(np.abs(vec))

    print("Sanity check: Riemannian mean of identical matrices")
    print(f"  max|mean - true matrix| = {err1:.2e} (expect ~0)")
    print(f"  max|tangent vector of matrix at its own mean| = {err2:.2e} (expect ~0)")
    ok = err1 < 1e-6 and err2 < 1e-6
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    sanity_check()

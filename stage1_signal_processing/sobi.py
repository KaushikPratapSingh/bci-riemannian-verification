"""
sobi.py — Second-Order Blind Identification (Belouchrani et al., 1997),
implemented via Jacobi-rotation approximate joint diagonalization of
time-lagged covariance matrices.
"""

import numpy as np

def joint_diagonalize(matrices, n_iter=200, tol=1e-8):
    """
    Approximate joint diagonalization of a list of symmetric matrices
    via Jacobi rotations. Returns orthogonal V such that V^T M_k V is
    ~diagonal for every matrix M_k in the input list simultaneously.
    """
    n = matrices[0].shape[0]
    V = np.eye(n)
    M = [m.copy() for m in matrices]

    for _ in range(n_iter):
        off_diag_energy = 0.0
        for p in range(n - 1):
            for q in range(p + 1, n):
                g = np.zeros((len(M), 2))
                for k, Mk in enumerate(M):
                    g[k, 0] = Mk[p, p] - Mk[q, q]
                    g[k, 1] = Mk[p, q] + Mk[q, p]
                G = g.T @ g
                # Minimise residual off-diagonal energy -> smallest eigenvalue
                eigvals, eigvecs = np.linalg.eigh(G)
                x, y = eigvecs[:, np.argmin(eigvals)]
                if np.sqrt(x * x + y * y) < 1e-12:
                    continue
                theta = 0.5 * np.arctan2(x, y)
                c, s = np.cos(theta), np.sin(theta)
                if abs(s) < 1e-12:
                    continue
                off_diag_energy += s ** 2

                J = np.eye(n)
                J[p, p], J[q, q] = c, c
                J[p, q], J[q, p] = s, -s
                for k in range(len(M)):
                    M[k] = J.T @ M[k] @ J
                V = V @ J
        if off_diag_energy < tol:
            break
    return V


def sobi(X, n_lags=20, n_components=None):
    """
    X: (n_samples, n_channels). 
    Returns (S, A, W): estimated sources, mixing matrix, and unmixing matrix 
    such that S = (X - mean(X)) @ W.T.
    """
    n_samples, n_channels = X.shape
    if n_components is None:
        n_components = n_channels

    Xc = X - X.mean(axis=0, keepdims=True)

    # Whitening (PCA on the lag-0 covariance)
    cov0 = (Xc.T @ Xc) / n_samples
    eigvals, eigvecs = np.linalg.eigh(cov0)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    Wh = np.diag(1.0 / np.sqrt(np.clip(eigvals, 1e-12, None))) @ eigvecs.T
    Z = Xc @ Wh.T

    # Time-lagged covariance matrices of the whitened data
    lag_matrices = []
    for tau in range(1, n_lags + 1):
        Rk = (Z[:-tau, :].T @ Z[tau:, :]) / (n_samples - tau)
        lag_matrices.append(0.5 * (Rk + Rk.T))

    V = joint_diagonalize(lag_matrices)
    S = Z @ V
    W = V.T @ Wh
    A = np.linalg.pinv(W)
    return S[:, :n_components], A, W
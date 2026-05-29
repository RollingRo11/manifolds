"""Directional Regression (Li & Wang 2007).

  M_DR = Σ_b (n_b/N) [(Σ - Σ_b)² + c · (μ_b - μ)(μ_b - μ)^T]

where Σ_b is within-bin covariance, μ_b is within-bin mean, c is a balance
constant (default: tr(M_SAVE)/tr(M_SIR) so the two terms contribute
comparable scale to the eigendecomposition).

The first term is SAVE (slice covariance variation — captures higher-order
structure like rotation and curvature). The second term is the SIR scatter
matrix (slice mean variation — captures monotone/first-moment structure
like helical phase). Combining them recovers manifolds that pure SAVE or
pure SIR each miss.

Same thin-SVD trick as save_recovery.py for tractability in 5120-D.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DRResult:
    embedding: np.ndarray             # [N, 1+d_M]
    V: np.ndarray                     # [d, 1+d_M]
    eigvals: np.ndarray               # eigenvalues of M_DR (descending)
    save_eigvals: np.ndarray          # eigvals of SAVE matrix alone
    sir_eigvals: np.ndarray           # eigvals of SIR scatter alone
    c_balance: float                  # the calibration constant used
    s: np.ndarray
    bin_pop: np.ndarray


def _thin_svd_basis(X: np.ndarray):
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Qt = np.linalg.svd(Xc, full_matrices=False)
    keep = S > S.max() * 1e-10
    Q = Qt[keep].T            # [d, K]
    A = Xc @ Q
    return Q, A


def dr_recovery(X: np.ndarray, w: np.ndarray, n_slices: int = 10,
                d_M: int = 2, c_balance: float | None = None) -> DRResult:
    X = X.astype(np.float64)
    w = w.astype(np.float64); w = w / (np.linalg.norm(w) + 1e-12)
    N, d = X.shape
    s = X @ w

    # Work in the data span
    Q, A = _thin_svd_basis(X)             # A: [N, K]; Q: [d, K]
    K = A.shape[1]
    Sigma_A = (A.T @ A) / N
    mu_A = A.mean(axis=0)                 # zero by construction

    edges = np.quantile(s, np.linspace(0, 1, n_slices + 1))
    bin_idx = np.clip(np.digitize(s, edges[1:-1]), 0, n_slices - 1)
    bin_pop = np.zeros(n_slices, dtype=int)

    M_save = np.zeros((K, K))
    M_sir = np.zeros((K, K))
    for k in range(n_slices):
        mask = bin_idx == k
        nk = int(mask.sum())
        bin_pop[k] = nk
        if nk < 2:
            continue
        Ak = A[mask]
        mu_k = Ak.mean(axis=0)
        Akc = Ak - mu_k
        Sigma_k = (Akc.T @ Akc) / nk
        # SAVE term: (Σ - Σ_k)²
        diff = Sigma_A - Sigma_k
        M_save += (nk / N) * (diff @ diff)
        # SIR term: (μ_k - μ)(μ_k - μ)^T
        delta = mu_k - mu_A
        M_sir += (nk / N) * np.outer(delta, delta)

    # Calibration
    tr_save = float(np.trace(M_save))
    tr_sir = float(np.trace(M_sir))
    if c_balance is None:
        c_balance = tr_save / max(tr_sir, 1e-12)

    M_dr = M_save + c_balance * M_sir

    # Eigendecompose all three for diagnostics
    save_eigs = np.linalg.eigvalsh(M_save)[::-1]
    sir_eigs = np.linalg.eigvalsh(M_sir)[::-1]
    eigvals, eigvecs = np.linalg.eigh(M_dr)
    order = np.argsort(-eigvals)
    eigvals = eigvals[order]; eigvecs = eigvecs[:, order]

    # Lift back to ambient
    dir_full = Q @ eigvecs[:, :d_M]
    dir_full /= (np.linalg.norm(dir_full, axis=0, keepdims=True) + 1e-12)
    V = np.column_stack([w, dir_full])
    embedding = X @ V

    return DRResult(
        embedding=embedding, V=V, eigvals=eigvals,
        save_eigvals=save_eigs, sir_eigvals=sir_eigs,
        c_balance=float(c_balance), s=s, bin_pop=bin_pop,
    )

"""SAVE — Sliced Average Variance Estimation (Cook & Weisberg, 1991).

  Inputs: probe w, activations X [N, d], target manifold dim d_M.
  1.  s = X @ w                                                   (probe score)
  2.  Bin into K equal-quantile slices: bin_idx[i]
  3.  Σ   = total covariance of X
      Σ_k = within-bin covariance of slice k
      M_SAVE = (1/N) Σ_k n_k (Σ - Σ_k)²
  4.  Top d_M eigenvectors of M_SAVE, stacked with w, give V [d, d_M+1].
  5.  Embedding:  Y = X @ V   shape [N, d_M+1].

Probe-faithfulness is a direct property of the algorithm: if w doesn't
organize the data, every Σ_k ≈ Σ → M_SAVE ≈ 0 → top eigenvalues collapse
to noise floor. The eigenvalue spectrum *is* a faithfulness diagnostic.

For tractability with d=5120, we work in the (much smaller) span of the
data X by first computing a thin SVD of X. The SAVE matrix and its
eigendecomposition are then O(min(N,d)³) instead of O(d³).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SAVEResult:
    embedding: np.ndarray             # [N, 1 + d_M]
    V: np.ndarray                     # [d, 1 + d_M] basis (col 0 = ŵ, then SAVE dirs)
    eigvals: np.ndarray               # eigenvalues of SAVE matrix (descending)
    save_dirs: np.ndarray             # [d, d_M] SAVE eigenvectors in ambient space
    s: np.ndarray                     # probe scores
    bin_pop: np.ndarray               # samples per slice


def _thin_svd_basis(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centered X = U S Q^T. Returns Q (basis spanning data), the projection
    matrix P (Q^T) and the in-basis representation A = X_centered @ Q."""
    Xc = X - X.mean(axis=0, keepdims=True)
    # economy SVD: only first min(N,d) singular values/vectors
    U, S, Qt = np.linalg.svd(Xc, full_matrices=False)
    # Keep only non-trivial singular values
    keep = S > S.max() * 1e-10
    Q = Qt[keep].T            # [d, K]
    A = Xc @ Q                 # [N, K]
    return Q, Qt[keep], A


def save_recovery(X: np.ndarray, w: np.ndarray, n_slices: int = 10,
                  d_M: int = 2) -> SAVEResult:
    X = X.astype(np.float64)
    w = w.astype(np.float64); w = w / (np.linalg.norm(w) + 1e-12)
    N, d = X.shape

    s = X @ w

    # Work in the data span: X_centered = U S Q^T. Operate in K-dim where
    # K = rank(X_centered) ≤ min(N, d).
    Q, Qt, A = _thin_svd_basis(X)            # A: [N, K]; Q: [d, K]

    # Total covariance in basis: Σ_A = (A.T A)/N
    Sigma_A = (A.T @ A) / N

    edges = np.quantile(s, np.linspace(0, 1, n_slices + 1))
    bin_idx = np.clip(np.digitize(s, edges[1:-1]), 0, n_slices - 1)
    bin_pop = np.zeros(n_slices, dtype=int)

    K = A.shape[1]
    M = np.zeros((K, K))
    for k in range(n_slices):
        mask = bin_idx == k
        nk = int(mask.sum())
        bin_pop[k] = nk
        if nk < 2:
            continue
        Ak = A[mask] - A[mask].mean(axis=0, keepdims=True)
        Sigma_k = (Ak.T @ Ak) / nk
        diff = Sigma_A - Sigma_k
        M += (nk / N) * (diff @ diff)

    # Top d_M eigenvectors of M (in basis A coordinates)
    eigvals, eigvecs = np.linalg.eigh(M)
    order = np.argsort(-eigvals)
    eigvals = eigvals[order]; eigvecs = eigvecs[:, order]

    # Lift back to ambient space: directions = Q @ eigvecs
    save_dirs_full = Q @ eigvecs[:, :d_M]                # [d, d_M]
    save_dirs_full /= (np.linalg.norm(save_dirs_full, axis=0, keepdims=True) + 1e-12)

    # Build V: probe + SAVE dirs (note: SAVE dirs are not generally
    # orthogonal to w; for the embedding we just project onto each).
    V = np.column_stack([w, save_dirs_full])             # [d, 1+d_M]

    embedding = X @ V                                    # [N, 1+d_M]
    return SAVEResult(
        embedding=embedding, V=V, eigvals=eigvals,
        save_dirs=save_dirs_full, s=s, bin_pop=bin_pop,
    )

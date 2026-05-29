"""SIR recovery v7 — v6 (PCA-reduced) + kernel-weighted local PCA.

Replaces hard bin boundaries with Gaussian kernel weights centered at each
evaluation point along the probe-score axis. Each evaluation point uses
*all* samples but with locality-preserving weights, so local tangent
estimates use more effective samples than hard binning while remaining
probe-conditional.

  s        = (X_pca · w_pca)                          (probe coord, in 64-D)
  R        = X_pca - s · w_pca                          (residual)
  s_k      = K equally-spaced eval points in s-range
  weights_k_i = exp(-((s_i - s_k) / h)^2)               (Gaussian kernel)
  Σ_k      = Σ_i weights_k_i (R_i - μ_k_R)(R_i - μ_k_R)^T
  V_k      = top-n_partner eigenvectors of Σ_k
  embed_i  = ( s_i, V_{k(i)} · R_i )    where k(i) = nearest eval point to s_i

Probe-faithfulness preserved: weights are functions of (s_i - s_k), so a
useless probe (s constant) produces a degenerate weighting where every
eval point has the same weights → V_k is the same global axis at every
bin → embedding is degenerate.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def kernel_local_pca_sir(X: np.ndarray, w: np.ndarray, n_eval: int = 20,
                         n_partner: int = 2, n_pca_pre: int = 64,
                         bandwidth_factor: float = 1.5) -> np.ndarray:
    """Returns [N, 1+n_partner] embedding."""
    X = X.astype(np.float64)
    w = w.astype(np.float64); w = w / (np.linalg.norm(w) + 1e-12)

    # ----- pre-PCA reduction -----
    pca = PCA(n_components=min(n_pca_pre, X.shape[0] - 1, X.shape[1])).fit(X)
    Xp = pca.transform(X)                              # [N, K]
    K_pca = Xp.shape[1]
    w_reduced = pca.components_ @ w
    w_reduced /= np.linalg.norm(w_reduced) + 1e-12

    s = Xp @ w_reduced
    R = Xp - s[:, None] * w_reduced[None, :]

    # ----- evaluation points + bandwidth -----
    s_lo, s_hi = np.quantile(s, 0.025), np.quantile(s, 0.975)
    s_eval = np.linspace(s_lo, s_hi, n_eval)
    # Default bandwidth: span / n_eval × factor ⇒ each eval pt sees ~factor neighbors of width
    h = (s_hi - s_lo) / max(n_eval - 1, 1) * bandwidth_factor

    bin_eigs = np.zeros((n_eval, n_partner, K_pca))
    for k_i, s_k in enumerate(s_eval):
        weights = np.exp(-((s - s_k) / h) ** 2)
        wsum = weights.sum() + 1e-12
        mu_R = (weights[:, None] * R).sum(axis=0) / wsum
        Rc = R - mu_R
        # Weighted SVD: scale rows by sqrt(weights)
        Rc_w = np.sqrt(weights)[:, None] * Rc
        _, _, Vt = np.linalg.svd(Rc_w, full_matrices=False)
        bin_eigs[k_i] = Vt[:n_partner]

    # Sign-align across eval points
    for k_i in range(1, n_eval):
        for j in range(n_partner):
            if bin_eigs[k_i, j] @ bin_eigs[k_i - 1, j] < 0:
                bin_eigs[k_i, j] *= -1

    # Build per-activation embedding: assign each i to its nearest eval point
    coords = np.zeros((len(X), n_partner))
    for i in range(len(X)):
        k_nearest = int(np.argmin(np.abs(s_eval - s[i])))
        coords[i] = bin_eigs[k_nearest] @ R[i]
    return np.column_stack([s, coords])

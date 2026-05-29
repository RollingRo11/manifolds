"""SIR recovery v4 — formal Sliced Inverse Regression (Li 1991).

  s        = (X @ w) / ||w||                            # probe coord
  R        = X - s · ŵ                                   # residual
  Bin R by s into K bins; compute weighted bin means μ_k of R.
  Σ_between = Σ_k (n_k / N) (μ_k - μ̄)(μ_k - μ̄)^T          # between-bin scatter
  Top-k eigenvectors of Σ_between are the SIR directions.
  embed    = ( s, V_1·R, ..., V_k·R )

Why this preserves probe-faithfulness AND lowers Procrustes vs v1:

  - Probe-faithfulness: Σ_between is the variance of bin means, which is only
    nontrivial if the BINNING organizes the data. For a random probe, bins
    are random subsets and μ_k ≈ μ̄ for all k, so Σ_between ≈ 0 — partner
    axes become noise eigenvectors. For an informative probe, μ_k traces a
    curve and Σ_between has clear leading eigenvectors aligned with the
    manifold's principal directions.

  - Sample efficiency: per-bin PCA uses only the n_k within-bin samples
    (here ~21). Σ_between uses all N samples (here 427) to estimate the
    K bin means, then takes their scatter matrix. Much lower estimation
    variance than per-bin local PCA.

  - The partner axes are global directions in X-space, like supervised PCA's
    PCs. So the recovery basis is comparable to supervised PCA's basis,
    which is what Procrustes wants.
"""
from __future__ import annotations

import numpy as np


def sir_proper(X: np.ndarray, w: np.ndarray, n_bins: int = 20,
               n_partner: int = 2, min_per_bin: int = 5) -> np.ndarray:
    """Returns [N, 1+n_partner] embedding."""
    X = X.astype(np.float64)
    w = w.astype(np.float64); w = w / (np.linalg.norm(w) + 1e-12)
    s = X @ w
    R = X - s[:, None] * w[None, :]

    # bin and compute bin means
    edges = np.quantile(s, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.digitize(s, edges[1:-1]), 0, n_bins - 1)

    overall_mean = R.mean(axis=0)
    valid_means = []
    valid_weights = []
    for k in range(n_bins):
        mask = idx == k
        nk = int(mask.sum())
        if nk < min_per_bin:
            continue
        mu_k = R[mask].mean(axis=0)
        valid_means.append(mu_k)
        valid_weights.append(nk)

    means = np.array(valid_means)                                  # [K_v, d]
    weights = np.array(valid_weights, dtype=np.float64)
    N_total = weights.sum()

    # weighted between-bin scatter (covariance of bin means)
    centered = means - overall_mean
    # Σ_between = Σ_k (n_k / N) c_k c_k^T
    # Compute via SVD of the row-weighted matrix:
    #   M = sqrt(n_k/N) * centered_k    (rows scaled)
    #   Σ_between = M^T @ M
    M = np.sqrt(weights / N_total)[:, None] * centered             # [K_v, d]
    # SVD: M = U S V^T, so M^T M = V S^2 V^T → eigenvectors of Σ_between are V
    _, S_v, Vt = np.linalg.svd(M, full_matrices=False)
    V = Vt[:n_partner]                                              # [n_partner, d]

    coords = R @ V.T
    return np.column_stack([s, coords])

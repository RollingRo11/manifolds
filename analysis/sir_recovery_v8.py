"""SIR recovery v8 — probe-conditional subspace + v1's per-bin PCA inside it.

  Step 1.  Run SIR-formal to find the probe-conditional subspace:
              μ_k = bin-mean of residuals; weighted between-bin scatter.
              Top-K_sub eigenvectors form a probe-conditional subspace.
  Step 2.  Project residuals into this subspace.
  Step 3.  Per-bin local PCA in the (now low-dim) subspace.
  Step 4.  Sign-align and stitch tangents.

The probe drives BOTH the binning AND the choice of subspace. Both signals
collapse to noise for a random probe, so probe-faithfulness is preserved
*more* than v1, not less. The subspace dimension (typically 16–32) gives
per-bin PCA much better signal-to-noise than the full 5120-D ambient.
"""
from __future__ import annotations

import numpy as np


def sir_subspace_per_bin(X: np.ndarray, w: np.ndarray, n_bins: int = 20,
                         n_partner: int = 2, k_subspace: int = 24,
                         min_per_bin: int = 4) -> np.ndarray:
    X = X.astype(np.float64)
    w = w.astype(np.float64); w = w / (np.linalg.norm(w) + 1e-12)
    N = X.shape[0]
    s = X @ w
    R = X - s[:, None] * w[None, :]

    edges = np.quantile(s, np.linspace(0, 1, n_bins + 1))
    bin_idx = np.clip(np.digitize(s, edges[1:-1]), 0, n_bins - 1)

    # ------- Step 1: SIR-formal subspace from between-bin scatter -------
    overall_mean = R.mean(axis=0)
    means = []
    weights = []
    for k in range(n_bins):
        m = bin_idx == k
        if m.sum() < min_per_bin:
            continue
        means.append(R[m].mean(axis=0))
        weights.append(int(m.sum()))
    means = np.array(means)
    weights = np.array(weights, dtype=np.float64)
    centered = means - overall_mean
    M = np.sqrt(weights / weights.sum())[:, None] * centered    # [K_v, d]
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    K_sub = min(k_subspace, Vt.shape[0])
    subspace_basis = Vt[:K_sub]                                 # [K_sub, d]

    # ------- Step 2: project residuals into probe-conditional subspace -------
    R_sub = R @ subspace_basis.T                                # [N, K_sub]

    # ------- Step 3: v1-style per-bin local PCA in the subspace -------
    bin_eigs = np.zeros((n_bins, n_partner, K_sub))
    valid = np.zeros(n_bins, dtype=bool)
    for k in range(n_bins):
        m = bin_idx == k
        if m.sum() < max(min_per_bin, n_partner + 1):
            continue
        Rk = R_sub[m] - R_sub[m].mean(axis=0)
        _, _, Vt2 = np.linalg.svd(Rk, full_matrices=False)
        bin_eigs[k] = Vt2[:n_partner]
        valid[k] = True

    # ------- Step 4: sign-align and stitch -------
    valid_idx = np.where(valid)[0]
    for vi, k in enumerate(valid_idx):
        if vi == 0: continue
        kp = valid_idx[vi - 1]
        for j in range(n_partner):
            if bin_eigs[k, j] @ bin_eigs[kp, j] < 0:
                bin_eigs[k, j] *= -1

    # Build embedding
    coords = np.zeros((N, n_partner))
    for i in range(N):
        k = bin_idx[i]
        if valid[k]:
            coords[i] = bin_eigs[k] @ R_sub[i]
        else:
            d_to_valid = np.abs(valid_idx - k)
            kp = valid_idx[int(np.argmin(d_to_valid))]
            coords[i] = bin_eigs[kp] @ R_sub[i]

    return np.column_stack([s, coords])

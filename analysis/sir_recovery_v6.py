"""SIR recovery v6 — v1 in a pre-PCA-reduced subspace.

Same probe-driven binning + per-bin local PCA as v1, but operate in the
top-K PCA subspace of X rather than the full d-dim space. With ~20 samples
per bin in 5120-dim, v1's per-bin SVD picks up a leading eigenvector that
is ~half signal, ~half random direction in the unused subspace. Reducing to
64 dims first concentrates samples relative to dimensionality and the
leading eigenvector becomes (much closer to) the actual local tangent.

The probe direction is projected into the reduced subspace, so probe-driven
binning is preserved exactly. This is *not* a global-PCA replacement (v3) —
it's v1 with a denoising preprocessor.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def reduced_per_bin_sir(X: np.ndarray, w: np.ndarray, n_bins: int = 20,
                        n_partner: int = 2, n_pca_pre: int = 64,
                        min_per_bin: int = 4) -> np.ndarray:
    """Returns [N, 1+n_partner] embedding."""
    X = X.astype(np.float64)
    w = w.astype(np.float64); w = w / (np.linalg.norm(w) + 1e-12)
    N = X.shape[0]

    # ------- pre-PCA reduction (denoising step) -------
    pca = PCA(n_components=min(n_pca_pre, X.shape[0] - 1, X.shape[1])).fit(X)
    Xp = pca.transform(X)                         # [N, K]
    Xp_centered = Xp                              # PCA already centered
    # Project the probe into the reduced subspace
    w_reduced = pca.components_ @ w               # [K]
    w_reduced /= np.linalg.norm(w_reduced) + 1e-12

    # ------- v1 in reduced space -------
    s = Xp_centered @ w_reduced
    R = Xp_centered - s[:, None] * w_reduced[None, :]

    edges = np.quantile(s, np.linspace(0, 1, n_bins + 1))
    bin_idx = np.clip(np.digitize(s, edges[1:-1]), 0, n_bins - 1)

    K_pca = Xp.shape[1]
    bin_eigs = np.zeros((n_bins, n_partner, K_pca))
    valid = np.zeros(n_bins, dtype=bool)
    for k in range(n_bins):
        m = bin_idx == k
        if m.sum() < max(min_per_bin, n_partner + 1):
            continue
        Rk = R[m] - R[m].mean(axis=0)
        _, _, Vt = np.linalg.svd(Rk, full_matrices=False)
        bin_eigs[k] = Vt[:n_partner]
        valid[k] = True

    # Sign-align consecutive bins
    valid_idx = np.where(valid)[0]
    for vi, k in enumerate(valid_idx):
        if vi == 0: continue
        kp = valid_idx[vi - 1]
        for j in range(n_partner):
            if bin_eigs[k, j] @ bin_eigs[kp, j] < 0:
                bin_eigs[k, j] *= -1

    # Build per-activation embedding
    coords = np.zeros((N, n_partner))
    for i in range(N):
        k = bin_idx[i]
        if valid[k]:
            coords[i] = bin_eigs[k] @ R[i]
        else:
            d_to_valid = np.abs(valid_idx - k)
            kp = valid_idx[int(np.argmin(d_to_valid))]
            coords[i] = bin_eigs[kp] @ R[i]

    return np.column_stack([s, coords])

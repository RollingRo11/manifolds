"""SIR recovery v5 — v1 (per-bin PCA) + smoothed tangents across bins.

Same pipeline as v1, but after sign-aligning per-bin eigenvectors we
Gaussian-smooth them across bin index (and re-orthonormalize). Reduces
per-bin estimation noise without iterating away from the probe-driven
binning.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d


def smoothed_per_bin_sir(X: np.ndarray, w: np.ndarray, n_bins: int = 20,
                         n_partner: int = 2, smooth_sigma: float = 2.0,
                         min_per_bin: int = 4) -> np.ndarray:
    """Returns [N, 1+n_partner] embedding."""
    X = X.astype(np.float64)
    w = w.astype(np.float64); w = w / (np.linalg.norm(w) + 1e-12)
    N, d = X.shape
    s = X @ w
    R = X - s[:, None] * w[None, :]

    edges = np.quantile(s, np.linspace(0, 1, n_bins + 1))
    bin_idx = np.clip(np.digitize(s, edges[1:-1]), 0, n_bins - 1)

    bin_eigs = np.zeros((n_bins, n_partner, d))
    valid = np.zeros(n_bins, dtype=bool)
    for k in range(n_bins):
        m = bin_idx == k
        if m.sum() < max(min_per_bin, n_partner + 1):
            continue
        Rk = R[m] - R[m].mean(axis=0)
        _, _, Vt = np.linalg.svd(Rk, full_matrices=False)
        bin_eigs[k] = Vt[:n_partner]
        valid[k] = True

    # Sign-align consecutively
    valid_idx = np.where(valid)[0]
    for vi, k in enumerate(valid_idx):
        if vi == 0: continue
        kp = valid_idx[vi - 1]
        for j in range(n_partner):
            if bin_eigs[k, j] @ bin_eigs[kp, j] < 0:
                bin_eigs[k, j] *= -1

    # Smooth across bins (closed=False here since we don't auto-detect closure)
    flat = bin_eigs.reshape(n_bins, -1)
    flat = gaussian_filter1d(flat, sigma=smooth_sigma, axis=0, mode="nearest")
    bin_eigs = flat.reshape(n_bins, n_partner, d)

    # Re-orthonormalize each bin's eigvec set (smoothing breaks orthogonality)
    for k in range(n_bins):
        Q, _ = np.linalg.qr(bin_eigs[k].T)
        bin_eigs[k] = Q.T

    # Build embedding
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

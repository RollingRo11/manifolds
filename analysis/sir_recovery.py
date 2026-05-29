"""Sliced-Inverse-Regression-style manifold recovery from one probe.

Algorithm (deterministic, given probe w):

    1. s_i = (w · h_i) / ||w||                           # 1D probe coord
    2. Bin activations by s into K equal-quantile bins.
    3. For each bin k:
        a. project residuals onto the (d-1)-dim subspace orthogonal to w:
               R_i = h_i - s_i · ŵ
        b. take top-n_partner principal components of R within bin k
           → e_1^{(k)}, ..., e_{n_partner}^{(k)}
    4. Sign-align eigenvectors across bins:
           if e_j^{(k)} · e_j^{(k-1)} < 0 then flip e_j^{(k)} → -e_j^{(k)}.
       For a *closed* manifold (e.g., a circle), the sign-aligned tangent
       parallel-transported around the loop should return to itself; if not,
       there is a residual orientation defect (Mobius-like) that we report.
    5. Build a (1 + n_partner)-D embedding:
           y_i = ( s_i, e_1^{(bin_i)} · R_i, e_2^{(bin_i)} · R_i, ... )
       Optionally, also report the "skeleton": one point per bin at the
       bin-conditional mean projected onto its own tangent directions.

This is Sliced Inverse Regression (Li 1991) with a probe score as the
"response," used to recover *manifold* tangents rather than a regression
subspace. With a probe, we skip the bootstrap-style iteration of Hastie &
Stuetzle's Principal Curves: the probe anchors the first projection.

Why this should work for a Monday probe on a 7-day cycle:
  - s ≈ cos(θ_h - θ_Mon)
  - Two activations at equal probe-score (e.g. Tue and Sun) lie on opposite
    sides of the chord through Monday. Within their bin, the direction
    separating them is sin(θ - θ_Mon) — the local tangent. PCA picks it up.
  - As s sweeps from +1 (near Monday) to −1 (Thursday-ish), the local tangent
    rotates continuously. Stitching the rotating tangents recovers the circle.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SIRRecovery:
    s: np.ndarray                        # [N] probe score per activation
    bin_idx: np.ndarray                  # [N] bin index 0..K-1
    bin_centers: np.ndarray              # [K] s value at the center of each bin
    bin_eigs: np.ndarray                 # [K, n_partner, d] sign-aligned local tangents
    bin_means: np.ndarray                # [K, d] mean activation per bin
    bin_pop: np.ndarray                  # [K] sample count per bin
    embedding: np.ndarray                # [N, 1+n_partner] per-activation embedding
    skeleton: np.ndarray                 # [K, 1+n_partner] one point per bin
    parallel_transport_defect: float     # closure of stitched tangents (closed manifold)


def slice_and_local_pca(
    X: np.ndarray, w: np.ndarray,
    n_bins: int = 20, n_partner: int = 2,
    min_per_bin: int = 6,
) -> SIRRecovery:
    X = X.astype(np.float64)
    w = w.astype(np.float64); w = w / (np.linalg.norm(w) + 1e-12)
    N, d = X.shape

    s = X @ w
    R = X - s[:, None] * w[None, :]

    edges = np.quantile(s, np.linspace(0, 1, n_bins + 1))
    bin_idx = np.clip(np.digitize(s, edges[1:-1]), 0, n_bins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    bin_eigs = np.zeros((n_bins, n_partner, d))
    bin_means = np.zeros((n_bins, d))
    bin_pop = np.zeros(n_bins, dtype=np.int64)
    eig_valid = np.zeros(n_bins, dtype=bool)
    for k in range(n_bins):
        mask = bin_idx == k
        bin_pop[k] = int(mask.sum())
        if bin_pop[k] < max(min_per_bin, n_partner + 1):
            continue
        Rk = R[mask]
        bin_means[k] = X[mask].mean(axis=0)
        Rk_c = Rk - Rk.mean(axis=0)
        _, S_v, Vt = np.linalg.svd(Rk_c, full_matrices=False)
        # eigenvectors lie in the (d-1)-dim subspace orthogonal to w by
        # construction, since R was already projected.
        bin_eigs[k] = Vt[:n_partner]
        eig_valid[k] = True

    # --- Sign-alignment across bins (Li 1991-style stitching) ---
    # Walk forward through valid bins; flip each eigenvector if its inner
    # product with the previous bin's same-rank eigenvector is negative.
    valid_idx = np.where(eig_valid)[0]
    for vi, k in enumerate(valid_idx):
        if vi == 0: continue
        k_prev = valid_idx[vi - 1]
        for j in range(n_partner):
            if bin_eigs[k, j] @ bin_eigs[k_prev, j] < 0:
                bin_eigs[k, j] *= -1

    # Closure check (parallel-transport defect) for closed manifolds: the
    # tangent parallel-transported around the loop should match the start.
    # Approximate "closure" by comparing first and last valid bins' first eig.
    if len(valid_idx) >= 2:
        first = bin_eigs[valid_idx[0], 0]
        last = bin_eigs[valid_idx[-1], 0]
        # for an open manifold (line), we'd expect last ≈ first;
        # for a closed cycle, last ≈ -first (sign flip after parallel transport
        # around a chord-bisected cycle, since we crossed the chord midpoint).
        defect = float(np.linalg.norm(last - first))
    else:
        defect = float("nan")

    # --- Build per-activation embedding ---
    coords = np.zeros((N, n_partner))
    for i in range(N):
        k = bin_idx[i]
        if eig_valid[k]:
            coords[i] = bin_eigs[k] @ R[i]   # (n_partner,)
        else:
            # Fallback: nearest valid bin
            d_to_valid = np.abs(valid_idx - k)
            k2 = valid_idx[int(np.argmin(d_to_valid))]
            coords[i] = bin_eigs[k2] @ R[i]
    embedding = np.column_stack([s, coords])

    # --- Skeleton (per-bin means in the embedding) ---
    skeleton = np.zeros((n_bins, 1 + n_partner))
    for k in range(n_bins):
        if eig_valid[k]:
            skeleton[k, 0] = centers[k]
            R_mean = bin_means[k] - s[bin_idx == k].mean() * w  # subtract w-component of bin mean
            skeleton[k, 1:] = bin_eigs[k] @ R_mean
        else:
            skeleton[k] = np.nan

    return SIRRecovery(
        s=s, bin_idx=bin_idx, bin_centers=centers,
        bin_eigs=bin_eigs, bin_means=bin_means, bin_pop=bin_pop,
        embedding=embedding, skeleton=skeleton,
        parallel_transport_defect=defect,
    )


# Convenience wrappers ------------------------------------------------------- #


def binary_probe_diff_of_means(X: np.ndarray, pos_mask: np.ndarray) -> np.ndarray:
    """Diff-of-means probe: mean(positive class) - mean(negative class)."""
    pos = X[pos_mask].mean(axis=0)
    neg = X[~pos_mask].mean(axis=0)
    w = pos - neg
    return w / (np.linalg.norm(w) + 1e-12)

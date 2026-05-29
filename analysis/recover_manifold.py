"""Recover the manifold a probe lies on.

Algorithm (probe-conditioned manifold reconstruction):

  Given activations X in R^d and a probe direction w in R^d:

  1.  s_i = w · h_i / ||w||                   probe scalar coordinate
  2.  R_i = h_i - s_i * (w / ||w||)            orthogonal residual
  3.  Run unsupervised manifold discovery (PCA + diffusion map) on {R_i}.
      Estimate intrinsic dim of R; pick the leading k-1 diffusion coords.
  4.  Recovered embedding Y_i = (s_i, dm_1(R_i), ..., dm_{k-1}(R_i)).

  Intuition: for a 1D manifold M ⊂ R^d and a probe w that is a chord through
  M (like a diff-of-means probe between two concepts on a cycle), s alone
  cannot distinguish points on the two sides of the chord — they have the
  same s. But after subtracting the probe component, the orthogonal residual
  still encodes which side: it lives on a 1D curve obtained by projecting M
  onto the (d-1)-dim hyperplane normal to w.

  For higher-dim manifolds, the orthogonal residuals carry the remaining
  intrinsic dimensions; we extract them with diffusion map.

The output is supervised-free: only `w` was used. We then validate against
held-out concept labels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from manifold import (
    diffusion_map,
    estimate_tangents,
    knn_geodesic_distances,
    local_pca_eigengap,
    persistence_diagrams,
    reduce_pca,
    two_nn_intrinsic_dim,
)


@dataclass
class RecoveredManifold:
    s: np.ndarray                       # [N] probe coord
    embedding: np.ndarray               # [N, k] full recovered embedding (s + orth coords)
    R_pca_explained: np.ndarray         # explained variance of orthogonal residuals
    intrinsic_dim_R: int                # estimated intrinsic dim of R
    diffusion_eigvals: np.ndarray
    diffusion_eigvecs: np.ndarray
    geodesic: np.ndarray                # [N, N] graph distances on R-manifold + s axis (mixed)
    h1_diagram: np.ndarray              # persistent H1 diagram
    intrinsic_dim_R_estimate: dict      # twonn + local-pca estimates
    partner_axis: np.ndarray | None = None  # extracted partner direction in original space


def find_partner_axes(R: np.ndarray, s: np.ndarray, n_axes: int = 2,
                       mid_quantile: float = 0.4) -> np.ndarray:
    """Find k orthogonal directions in R that have the widest spread at
    s ≈ s_median.

    Intuition: for a cyclic manifold and a chord probe w, R has a *bimodal*
    distribution conditional on s, with the two modes separated along the
    partner axis. The midpoint of s is where this bimodal spread is largest;
    PCA on R within the mid-s slab recovers the partner. The next axis is
    the second PC, etc. — these are mutually orthogonal partner directions.

    For non-cyclic manifolds (lines, curves, surfaces) this still picks up
    the dominant within-slab variation, which is the next intrinsic axis.
    """
    s_med = float(np.median(s))
    s_iqr = float(np.subtract(*np.quantile(s, [0.75, 0.25])))
    half = mid_quantile * s_iqr / 2 + 1e-9
    mask = np.abs(s - s_med) < half
    if mask.sum() < 20:
        mask = np.ones(R.shape[0], dtype=bool)
    Rm = R[mask] - R[mask].mean(axis=0)
    _, _, Vt = np.linalg.svd(Rm, full_matrices=False)
    return Vt[:n_axes]   # [n_axes, d]


def recover(X: np.ndarray, w: np.ndarray, k_pca: int = 64,
            k_nn: int = 20, n_eigs: int = 8,
            intrinsic_dim_cap: int = 4,
            partner_method: str = "mid_slab_pca",
            n_partner_axes: int = 2) -> RecoveredManifold:
    X = X.astype(np.float64)
    w = w.astype(np.float64); w = w / (np.linalg.norm(w) + 1e-12)
    s = X @ w
    R = X - s[:, None] * w[None, :]      # orthogonal residual in original space

    # Reduce R to manageable dim. Use enough components to capture ~all variance
    # but not so many we drown the manifold in noise.
    pca_R = reduce_pca(R, k_pca=k_pca)
    Rp = pca_R.X.astype(np.float64)

    # Intrinsic dim of the residual manifold
    id_two = two_nn_intrinsic_dim(Rp)
    _, id_local = local_pca_eigengap(Rp, k=min(30, len(Rp) // 4))
    intrinsic_dim_R = max(1, min(int(round(id_local)), intrinsic_dim_cap))

    # Diffusion map on residuals — kept for diagnostic, but we use a dedicated
    # partner-axis estimator below for the recovered embedding.
    dm = diffusion_map(Rp, k_nn=k_nn, n_eigs=n_eigs)

    partner_axis_full = None
    if partner_method == "mid_slab_pca":
        # Operate in the original residual space (not PCA-reduced) so the
        # returned partner_axes lives in the same coordinates as w.
        V = find_partner_axes(R, s, n_axes=n_partner_axes)  # [n_axes, d]
        # Project residuals onto these axes directly in original space:
        # coords[i, k] = V[k] · (h_i - s_i*ŵ).
        coords = R @ V.T   # [N, n_axes]
        partner_axis_full = V
    else:  # legacy: diffusion-map coords
        coords = dm.coords(n_dims=intrinsic_dim_R)

    # Standardize so probe coord and partner coord(s) have comparable scale
    s_std = (s - s.mean()) / (s.std() + 1e-9)
    if coords.std() > 0:
        coords_std = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-9)
    else:
        coords_std = coords
    embedding = np.concatenate([s_std[:, None], coords_std], axis=1)

    # kNN-graph geodesic on the recovered embedding (for distance comparison)
    geo = knn_geodesic_distances(embedding.astype(np.float32), k_nn=k_nn)

    res = persistence_diagrams(Rp, max_dim=1)
    h1 = res["dgms"][1]

    return RecoveredManifold(
        s=s, embedding=embedding,
        R_pca_explained=pca_R.explained,
        intrinsic_dim_R=intrinsic_dim_R,
        diffusion_eigvals=dm.eigvals,
        diffusion_eigvecs=dm.eigvecs,
        geodesic=geo,
        h1_diagram=h1,
        intrinsic_dim_R_estimate={"twonn": float(id_two),
                                   "local_pca_median": int(round(id_local))},
        partner_axis=partner_axis_full,
    )

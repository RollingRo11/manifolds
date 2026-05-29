"""SIR recovery v2 — bin-mean smoothing + Hastie-Stuetzle principal-curves iteration.

Improvements over v1:

  (A) Bin-mean smoothing.  Per-bin centroids and per-bin tangent eigenvectors
      are smoothed across adjacent bins with a Gaussian kernel before being
      used to produce the embedding.  Reduces per-bin estimation noise (we
      have ~20 samples per bin; raw per-bin PCA is high-variance).

  (B) Hastie-Stuetzle iteration.  After the initial probe-driven binning,
      build a piecewise-linear curve through the bin centroids in original
      activation space.  For each activation, project onto the curve and
      record arc-length τ.  Re-bin by τ (uniform along the curve), recompute
      bin means and tangents.  Iterate until the curve stops moving.

      Closure: if the manifold is detected as closed (last and first bin
      means are within k_close × the typical bin gap), wrap τ to [0, L) and
      re-bin uniformly along the closed curve.

The probe is used only for INITIALIZATION (Phase 1).  After Phase 2
iterates, the embedding is parameterized by τ (arc length along the
recovered curve), not by w·h directly — this removes the probe-axis bias
that hurts Procrustes alignment with supervised PCA.

Output embedding for each activation:
  y_i = (τ_i, e_1(τ_i) · (h_i - c(τ_i)), e_2(τ_i) · (h_i - c(τ_i)))
where c(τ) is the curve and e_1(τ), e_2(τ) are smoothed local tangent /
co-tangent directions at arc-length τ.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d


@dataclass
class HSRecovery:
    tau: np.ndarray                     # [N] arc-length parameter per activation
    bin_centers_tau: np.ndarray         # [K] tau at center of each bin
    bin_centroids: np.ndarray           # [K, d] curve points (smoothed bin means in X-space)
    bin_eigs: np.ndarray                # [K, n_partner, d] sign-aligned smoothed local tangents
    embedding: np.ndarray               # [N, 1+n_partner] (tau, partner coords)
    closed: bool
    iter_curve_drift: list              # max bin-centroid movement at each HS iteration
    parallel_transport_defect: float    # sign-aligned closure error


# ----- arc-length parameterization helpers --------------------------------- #


def piecewise_linear_curve_arclen(centroids: np.ndarray, closed: bool = False) -> np.ndarray:
    """Cumulative arc-length along a piecewise-linear curve through centroids.
    Returns array of length K with arc-length at each centroid (start = 0)."""
    K = centroids.shape[0]
    if closed:
        # Close the loop
        deltas = np.diff(np.vstack([centroids, centroids[:1]]), axis=0)
    else:
        deltas = np.diff(centroids, axis=0)
    seg = np.linalg.norm(deltas, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    return cum[:K] if not closed else cum[:K]


def project_onto_piecewise_curve(X: np.ndarray, centroids: np.ndarray,
                                  closed: bool = False) -> np.ndarray:
    """For each point in X, find arc-length τ of its projection onto the
    piecewise-linear curve through centroids."""
    K = centroids.shape[0]
    if closed:
        cents_loop = np.vstack([centroids, centroids[:1]])  # [K+1, d]
    else:
        cents_loop = centroids
    M = cents_loop.shape[0] - 1  # number of segments
    # Segment parameterization: for each segment i, point p(t) = a_i + t * (b_i - a_i), t ∈ [0,1]
    a = cents_loop[:-1]                           # [M, d]
    b = cents_loop[1:]                            # [M, d]
    seg_vec = b - a                               # [M, d]
    seg_len2 = (seg_vec ** 2).sum(axis=1)         # [M]
    # cumulative arc-length up to start of each segment
    seg_len = np.sqrt(seg_len2)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])  # [M+1]

    # For each point and each segment, compute t* and squared dist
    # X: [N, d], a: [M, d], seg_vec: [M, d]
    # diff[n, m, d] = X[n] - a[m]
    # t[n, m] = (diff @ seg_vec[m]) / seg_len2[m]
    N = X.shape[0]
    # batched in chunks to limit memory: but with K=20 and N=400, 400*20*5120 = ~40M floats; OK
    diff = X[:, None, :] - a[None, :, :]                 # [N, M, d]
    t = (diff * seg_vec[None, :, :]).sum(axis=2) / (seg_len2[None, :] + 1e-12)
    t_clip = np.clip(t, 0.0, 1.0)
    # closest point on segment m to X[n]
    closest = a[None, :, :] + t_clip[:, :, None] * seg_vec[None, :, :]   # [N, M, d]
    dist2 = ((X[:, None, :] - closest) ** 2).sum(axis=2)                  # [N, M]
    best_seg = np.argmin(dist2, axis=1)                                   # [N]
    best_t = t_clip[np.arange(N), best_seg]
    tau = cum[best_seg] + best_t * seg_len[best_seg]
    return tau


def initial_bin_means(X: np.ndarray, w: np.ndarray, n_bins: int):
    s = X @ w
    edges = np.quantile(s, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.digitize(s, edges[1:-1]), 0, n_bins - 1)
    means = np.zeros((n_bins, X.shape[1]))
    for k in range(n_bins):
        m = idx == k
        if m.sum() == 0:
            continue
        means[k] = X[m].mean(axis=0)
    return s, idx, means


def smooth_curve(centroids: np.ndarray, sigma: float = 1.0,
                 closed: bool = False) -> np.ndarray:
    """1D Gaussian smoothing along the bin axis."""
    if closed:
        return gaussian_filter1d(centroids, sigma=sigma, axis=0, mode="wrap")
    return gaussian_filter1d(centroids, sigma=sigma, axis=0, mode="nearest")


def detect_closed(centroids: np.ndarray, k_close: float = 1.5) -> bool:
    """If the start and end of the curve are within k_close × median-bin-gap,
    treat it as closed."""
    deltas = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
    median_gap = float(np.median(deltas))
    closure_gap = float(np.linalg.norm(centroids[-1] - centroids[0]))
    return closure_gap < k_close * median_gap


# ----- main algorithm ------------------------------------------------------ #


def hs_principal_curve(X: np.ndarray, w: np.ndarray,
                       n_bins: int = 20, n_partner: int = 2,
                       n_iter: int = 5,
                       smooth_sigma: float = 1.0,
                       closed: bool | None = None,
                       verbose: bool = False) -> HSRecovery:
    X = X.astype(np.float64)
    w = w.astype(np.float64); w = w / (np.linalg.norm(w) + 1e-12)
    N, d = X.shape

    # Phase 1: probe-driven initialization
    s, bin_idx, centroids = initial_bin_means(X, w, n_bins)

    # Detect closure (if not specified by caller)
    if closed is None:
        closed = detect_closed(centroids)
    if verbose:
        print(f"  [HS] closed manifold detected = {closed}")

    iter_drift = []
    for it in range(n_iter):
        # Smooth current curve
        smoothed = smooth_curve(centroids, sigma=smooth_sigma, closed=closed)
        # Project all points onto the smoothed piecewise-linear curve
        tau = project_onto_piecewise_curve(X, smoothed, closed=closed)
        # Re-bin uniformly by τ
        if closed:
            L_total = piecewise_linear_curve_arclen(np.vstack([smoothed, smoothed[:1]]),
                                                     closed=False)[-1]
            tau_norm = tau % L_total
            edges = np.linspace(0, L_total, n_bins + 1)
        else:
            tau_norm = tau
            edges = np.quantile(tau_norm, np.linspace(0, 1, n_bins + 1))
        new_idx = np.clip(np.digitize(tau_norm, edges[1:-1]), 0, n_bins - 1)
        new_centroids = np.zeros_like(smoothed)
        for k in range(n_bins):
            m = new_idx == k
            if m.sum() < 2:
                new_centroids[k] = smoothed[k]
            else:
                new_centroids[k] = X[m].mean(axis=0)
        drift = float(np.linalg.norm(new_centroids - centroids, axis=1).mean())
        iter_drift.append(drift)
        if verbose:
            print(f"  [HS] iter {it+1}: mean drift = {drift:.4f}")
        centroids = new_centroids
        bin_idx = new_idx

    # Final smoothed curve
    final_curve = smooth_curve(centroids, sigma=smooth_sigma, closed=closed)
    tau = project_onto_piecewise_curve(X, final_curve, closed=closed)

    # Per-bin local PCA on residuals from the curve
    # Re-bin once more by final tau for stable per-bin grouping
    if closed:
        L_total = piecewise_linear_curve_arclen(np.vstack([final_curve, final_curve[:1]]),
                                                 closed=False)[-1]
        tau_norm = tau % L_total
        edges = np.linspace(0, L_total, n_bins + 1)
    else:
        tau_norm = tau
        edges = np.quantile(tau_norm, np.linspace(0, 1, n_bins + 1))
    bin_idx = np.clip(np.digitize(tau_norm, edges[1:-1]), 0, n_bins - 1)
    bin_centers_tau = 0.5 * (edges[:-1] + edges[1:])

    # Project residuals to remove the curve-direction component before PCA.
    # Use local segment direction as the "curve direction" at each bin.
    bin_eigs = np.zeros((n_bins, n_partner, d))
    valid = np.zeros(n_bins, dtype=bool)
    for k in range(n_bins):
        m = bin_idx == k
        if m.sum() < n_partner + 2:
            continue
        # local curve tangent at bin k (finite-difference between adjacent centroids)
        if closed:
            t_loc = final_curve[(k + 1) % n_bins] - final_curve[(k - 1) % n_bins]
        else:
            kp = min(k + 1, n_bins - 1); km = max(k - 1, 0)
            t_loc = final_curve[kp] - final_curve[km]
        t_loc = t_loc / (np.linalg.norm(t_loc) + 1e-12)
        Rk = X[m] - final_curve[k]                                    # offset from curve
        Rk = Rk - (Rk @ t_loc)[:, None] * t_loc[None, :]               # remove curve direction
        Rk_c = Rk - Rk.mean(axis=0)
        if Rk_c.shape[0] < 2: continue
        _, _, Vt = np.linalg.svd(Rk_c, full_matrices=False)
        bin_eigs[k] = Vt[:n_partner]
        valid[k] = True

    # Smooth eigenvectors across bins after sign-alignment.
    # First sign-align greedily, then enforce closure if applicable.
    valid_idx = np.where(valid)[0]
    for vi, k in enumerate(valid_idx):
        if vi == 0: continue
        kp = valid_idx[vi - 1]
        for j in range(n_partner):
            if bin_eigs[k, j] @ bin_eigs[kp, j] < 0:
                bin_eigs[k, j] *= -1

    if closed and len(valid_idx) >= 3:
        # Closure correction: distribute the closure defect evenly across bins
        first = bin_eigs[valid_idx[0], 0]
        last = bin_eigs[valid_idx[-1], 0]
        # if last ≈ +first ⇒ no flip; if last ≈ -first ⇒ flip cycle (we expect this for chord-bisected loops)
        # We do nothing for now; accept the natural sign pattern.
        # (Smoothing across the wrap will partially fix.)
        pass

    # Smooth eigenvectors with a small kernel so neighboring bins have continuous tangents.
    bin_eigs = smooth_curve(bin_eigs.reshape(n_bins, -1),
                            sigma=smooth_sigma, closed=closed).reshape(n_bins, n_partner, d)
    # Re-orthonormalize each bin's set of eigenvectors after smoothing
    for k in range(n_bins):
        Q, _ = np.linalg.qr(bin_eigs[k].T)  # d x n_partner
        bin_eigs[k] = Q.T

    # Closure defect for reporting
    if closed and len(valid_idx) >= 2:
        defect = float(np.linalg.norm(bin_eigs[valid_idx[-1], 0] - bin_eigs[valid_idx[0], 0]))
    else:
        defect = float("nan")

    # Build per-activation embedding (τ, e1·R_centered, e2·R_centered)
    coords = np.zeros((N, n_partner))
    for i in range(N):
        k = bin_idx[i]
        rk = X[i] - final_curve[k]
        coords[i] = bin_eigs[k] @ rk
    embedding = np.column_stack([tau_norm, coords])

    return HSRecovery(
        tau=tau_norm,
        bin_centers_tau=bin_centers_tau,
        bin_centroids=final_curve,
        bin_eigs=bin_eigs,
        embedding=embedding,
        closed=closed,
        iter_curve_drift=iter_drift,
        parallel_transport_defect=defect,
    )

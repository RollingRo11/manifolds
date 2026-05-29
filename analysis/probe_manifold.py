"""Probe-driven manifold: given a 1D contrastive probe direction w, build the
curve mu(s) = E[h | w.h ~ s] in activation space and characterize it.

This inverts the question from probe_tangent.py:
  probe_tangent.py: "given a manifold, does the probe align with it?"
  probe_manifold.py: "given a probe, what manifold does it parameterize?"

Outputs:
  curve              [B, d]   bin-mean activations (or KDE-smoothed centroids)
  s_grid             [B]      probe scores corresponding to each bin
  arclen, chord_len  scalars  with ratio = curvature_index
  tangent_at_bins    [B, d]   normalized mu'(s) (finite differences on spline)
  cos_tangent_to_w   [B]      alignment of local tangent with the probe chord
  residual_cov_eig   [B, k]   leading eigenvalues of residual cov per bin
  score_density      [B]      kde of p(s) at each bin center (for sanity)

Curvature ratio > 1 + small ==> probe direction is a chord across a curved
manifold. Curvature ratio ≈ 1 ==> linear-representation hypothesis holds for
this feature. Score density bimodal ==> "manifold" is two clusters bridged
by an extrapolated spline; treat the result with caution.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.stats import gaussian_kde


@dataclass
class ProbeManifold:
    s_bin_centers: np.ndarray   # [B]
    mu_curve: np.ndarray        # [B, d] — pre-spline bin means (in PCA space)
    spline_eval: np.ndarray     # [B_dense, d] — spline-resampled curve
    s_dense: np.ndarray         # [B_dense]
    arclen: float
    chord_len: float
    curvature_ratio: float
    tangent_at_bins: np.ndarray # [B, d] unit-norm
    cos_tangent_to_w: np.ndarray  # [B]
    residual_cov_eig: np.ndarray  # [B, k]
    residual_cov_top_dir: np.ndarray  # [B, d]  leading eigenvector at each bin
    score_density: np.ndarray   # [B] — kde at bin centers
    n_per_bin: np.ndarray       # [B]
    bimodality: float           # Hartigan-style dip on p(s); 0 = unimodal


def _equal_count_bins(s: np.ndarray, n_bins: int) -> np.ndarray:
    qs = np.linspace(0, 1, n_bins + 1)
    return np.quantile(s, qs)


def fit_probe_manifold(
    X: np.ndarray,
    w: np.ndarray,
    n_bins: int = 16,
    n_dense: int = 200,
    eig_k: int = 4,
    min_per_bin: int = 5,
) -> ProbeManifold:
    """X: [N, d] activations (typically PCA-reduced).
    w: [d] direction in the *same* basis as X (project before passing in)."""
    w = w.astype(np.float64)
    w = w / (np.linalg.norm(w) + 1e-12)
    s = X @ w  # [N]

    edges = _equal_count_bins(s, n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.digitize(s, edges[1:-1], right=False)  # 0..n_bins-1

    d = X.shape[1]
    mu = np.zeros((n_bins, d), dtype=np.float64)
    n_per_bin = np.zeros(n_bins, dtype=np.int64)
    res_eig = np.zeros((n_bins, eig_k), dtype=np.float64)
    res_top = np.zeros((n_bins, d), dtype=np.float64)
    for b in range(n_bins):
        mask = bin_idx == b
        n_per_bin[b] = int(mask.sum())
        if n_per_bin[b] >= 1:
            Xb = X[mask]
            mu[b] = Xb.mean(axis=0)
            if n_per_bin[b] >= max(min_per_bin, eig_k + 1):
                Xc = Xb - mu[b]
                # eigendecomposition of d×d cov via SVD on Xc (faster when N_b < d)
                _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
                lam = (S ** 2) / max(n_per_bin[b] - 1, 1)
                res_eig[b, : min(eig_k, len(lam))] = lam[: eig_k]
                res_top[b] = Vt[0]
        else:
            mu[b] = X.mean(axis=0)  # fallback
    # Spline through bin means (parameterized by bin center s value)
    spline = CubicSpline(centers, mu, axis=0, bc_type="natural")
    s_dense = np.linspace(centers[0], centers[-1], n_dense)
    pts = spline(s_dense)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arclen = float(seg.sum())
    chord_len = float(np.linalg.norm(pts[-1] - pts[0]))
    curvature_ratio = arclen / (chord_len + 1e-12)

    # Tangent at each bin center via spline derivative
    dmu = spline(centers, 1)  # first derivative
    tnorm = np.linalg.norm(dmu, axis=1, keepdims=True) + 1e-12
    tangent = dmu / tnorm
    cos_tw = tangent @ w  # [B]

    # KDE of probe scores at bin centers
    try:
        kde = gaussian_kde(s, bw_method="silverman")
        density = kde(centers)
    except Exception:
        density = n_per_bin.astype(float) / max(n_per_bin.sum(), 1)

    # Cheap unimodality check: dip = 1 - max(density) / mean(density);
    # >0.3 indicates strong bimodality. (Not a real Hartigan dip but proxies it.)
    bimodality = float(np.var(density) / (density.mean() ** 2 + 1e-12))

    return ProbeManifold(
        s_bin_centers=centers,
        mu_curve=mu.astype(np.float32),
        spline_eval=pts.astype(np.float32),
        s_dense=s_dense,
        arclen=arclen,
        chord_len=chord_len,
        curvature_ratio=curvature_ratio,
        tangent_at_bins=tangent.astype(np.float32),
        cos_tangent_to_w=cos_tw.astype(np.float32),
        residual_cov_eig=res_eig.astype(np.float32),
        residual_cov_top_dir=res_top.astype(np.float32),
        score_density=density.astype(np.float32),
        n_per_bin=n_per_bin,
        bimodality=bimodality,
    )


def random_curvature_baseline(X: np.ndarray, n_dirs: int = 64, n_bins: int = 12,
                                seed: int = 0) -> tuple[float, float]:
    """Mean +/- std of curvature_ratio for random unit directions in span(X).

    Tells us how much of an observed curvature is finite-sample bin-mean noise.
    """
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    ratios = []
    for _ in range(n_dirs):
        v = rng.standard_normal(d).astype(np.float64)
        v /= np.linalg.norm(v) + 1e-12
        pm = fit_probe_manifold(X, v, n_bins=n_bins)
        ratios.append(pm.curvature_ratio)
    return float(np.mean(ratios)), float(np.std(ratios))


def report(pm: ProbeManifold, name: str = "") -> str:
    cos_min, cos_max, cos_mean = float(pm.cos_tangent_to_w.min()), float(pm.cos_tangent_to_w.max()), float(pm.cos_tangent_to_w.mean())
    eig0_var = float(np.var(pm.residual_cov_eig[:, 0]) / (pm.residual_cov_eig[:, 0].mean() ** 2 + 1e-12))
    return (
        f"[probe-manifold {name}]\n"
        f"  curvature_ratio = arclen/chord = {pm.curvature_ratio:.3f}  "
        f"(>>1 ⇒ curved; ≈1 ⇒ LRH-compatible)\n"
        f"  tangent · w  : mean={cos_mean:.3f}  range=[{cos_min:.3f}, {cos_max:.3f}]  "
        f"(if mean<<1 the probe is a chord, not a tangent)\n"
        f"  residual cov leading-eig variation across bins (rel-var) = {eig0_var:.3f}  "
        f"(thickness varies along the manifold if >0)\n"
        f"  bimodality of p(s) = {pm.bimodality:.3f}  (>0.3 ⇒ disconnected clusters)\n"
        f"  per-bin counts: min={pm.n_per_bin.min()} median={int(np.median(pm.n_per_bin))} max={pm.n_per_bin.max()}"
    )

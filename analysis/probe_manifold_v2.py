"""Extended probe-driven manifold detectors.

Each function returns a single number (with optional auxiliary outputs) that
captures one kind of "the probe is part of a curved/nonlinear structure". They
are deliberately complementary: passing several together is much stronger
evidence than passing one.

  1. gauss_map_turning(pm)
        Total angular turning of the spline tangent along mu(s). 0 for a
        straight line; >0 for any bending. Robust to scale because it's
        normalized by chord length.

  2. moving_frame_rotation(X, w)
        At each bin, do local PCA. Track how the leading principal direction
        rotates as we walk along w. Returns total rotation angle and the
        direction of biggest rotation (a candidate "partner axis" the probe
        couples to).

  3. tube_anisotropy(X, w)
        Per-bin residual covariance: leading eigenvalue, second eigenvalue,
        anisotropy ratio. If anisotropy varies along the probe, the manifold
        thickness changes; if it's constant but >1, the cross-section is
        elongated in a consistent direction.

  4. counterfactual_smoothwalk(X, w)
        For each starting activation h0, walk along w in fixed steps. Measure
        how fast the path diverges from the data manifold (mean nearest-
        neighbor distance growth). If the probe is intrinsic, distance stays
        bounded; if not, it grows linearly with step count (teleportation).

  5. nullspace_nonlinear_R2(X, w)
        Train a non-linear regressor (here: kernel ridge with RBF) to predict
        s = w·h from the orthogonal projection h_perp. R² is the cleanest
        single-number test:
          R² ≈ 0  ⇒  probe is independent of orthogonal data → linear feature
          R² > 0  ⇒  activations satisfy a non-linear constraint coupling
                    w·h to other coordinates → probe lies on a manifold.

We also include a few utilities for randomization baselines so each detector
ships with a "what would a random direction give?" reference.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

from probe_manifold import ProbeManifold, fit_probe_manifold


# -- 1. Gauss-map turning ---------------------------------------------------- #


def gauss_map_turning(pm: ProbeManifold) -> dict:
    """Total angular turning of the spline tangent (in radians) along mu(s)."""
    # Tangents at dense sample points
    spline = CubicSpline(pm.s_bin_centers, pm.mu_curve, axis=0, bc_type="natural")
    s_dense = np.linspace(pm.s_bin_centers[0], pm.s_bin_centers[-1], 200)
    dmu = spline(s_dense, 1)
    norms = np.linalg.norm(dmu, axis=1, keepdims=True) + 1e-12
    T = dmu / norms
    # Angle between consecutive tangents
    cos = np.clip((T[:-1] * T[1:]).sum(axis=1), -1.0, 1.0)
    dtheta = np.arccos(cos)
    total_turning = float(dtheta.sum())
    # Normalized: turning per unit chord
    chord = float(np.linalg.norm(spline(s_dense[-1]) - spline(s_dense[0])) + 1e-12)
    return {
        "total_turning_rad": total_turning,
        "turning_per_unit_chord": total_turning / chord,
        "turning_per_unit_score_range": total_turning / max(s_dense[-1] - s_dense[0], 1e-9),
        "max_per_step_rad": float(dtheta.max()),
    }


# -- 2. Moving-frame rotation ------------------------------------------------ #


def moving_frame_rotation(X: np.ndarray, w: np.ndarray, n_bins: int = 12,
                          k_local: int = 30) -> dict:
    """Track how the leading local PC rotates as we walk along the probe.

    For each bin in s, take its k_local nearest neighbors in X (under the full
    metric) and do PCA on them. The leading eigenvector at each bin is u_b.
    The "rotation rate" is sum_b angle(u_b, u_{b+1}). If this rotation is
    significant and aligns with a fixed direction across all bins, that
    direction is the probe's *partner axis* — together with w, it spans a 2D
    manifold the probe lies on.
    """
    w = w / (np.linalg.norm(w) + 1e-12)
    s = X @ w
    edges = np.quantile(s, np.linspace(0, 1, n_bins + 1))
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.digitize(s, edges[1:-1])

    leaders = np.zeros((n_bins, X.shape[1]))
    have = np.zeros(n_bins, dtype=bool)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() < max(5, k_local // 2):
            continue
        Xb = X[mask]
        Xc = Xb - Xb.mean(axis=0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        leaders[b] = Vt[0]
        have[b] = True

    # Resolve sign flips so consecutive leaders point the "same way"
    for b in range(1, n_bins):
        if have[b] and have[b - 1] and leaders[b] @ leaders[b - 1] < 0:
            leaders[b] *= -1

    # Total rotation angle (radians) along the bin sequence
    angles = []
    for b in range(n_bins - 1):
        if have[b] and have[b + 1]:
            c = float(np.clip(leaders[b] @ leaders[b + 1], -1.0, 1.0))
            angles.append(float(np.arccos(c)))
    total_rot = float(np.sum(angles))

    # Candidate partner axis: the unit-norm component of (leaders[-1] - leaders[0])
    # orthogonalized against w. (If leaders rotate consistently, this points along
    # the dominant rotation plane's other axis.)
    if have[0] and have[-1]:
        delta = leaders[-1] - leaders[0]
        delta = delta - (delta @ w) * w
        partner = delta / (np.linalg.norm(delta) + 1e-12)
    else:
        partner = np.zeros_like(w)

    return {
        "total_rotation_rad": total_rot,
        "n_bins_with_data": int(have.sum()),
        "max_per_step_rad": float(max(angles)) if angles else 0.0,
        "partner_axis": partner,
        "leaders": leaders,
        "centers": centers,
    }


# -- 3. Tube anisotropy ----------------------------------------------------- #


def tube_anisotropy(X: np.ndarray, w: np.ndarray, n_bins: int = 12) -> dict:
    """Per-bin residual covariance leading-eig and anisotropy."""
    w = w / (np.linalg.norm(w) + 1e-12)
    s = X @ w
    edges = np.quantile(s, np.linspace(0, 1, n_bins + 1))
    bin_idx = np.digitize(s, edges[1:-1])

    lam1, lam2 = [], []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() < 6:
            lam1.append(np.nan); lam2.append(np.nan); continue
        Xb = X[mask]
        Xc = Xb - Xb.mean(axis=0)
        _, S, _ = np.linalg.svd(Xc, full_matrices=False)
        ev = (S ** 2) / max(mask.sum() - 1, 1)
        lam1.append(float(ev[0]))
        lam2.append(float(ev[1]) if len(ev) > 1 else np.nan)
    lam1 = np.array(lam1); lam2 = np.array(lam2)
    aniso = lam1 / (lam2 + 1e-12)
    return {
        "lam1_mean": float(np.nanmean(lam1)),
        "lam1_relvar": float(np.nanvar(lam1) / (np.nanmean(lam1) ** 2 + 1e-12)),
        "anisotropy_mean": float(np.nanmean(aniso)),
        "anisotropy_max": float(np.nanmax(aniso)),
        "lam1_per_bin": lam1,
        "lam2_per_bin": lam2,
    }


# -- 4. Counterfactual smooth-walk ------------------------------------------ #


def counterfactual_smoothwalk(X: np.ndarray, w: np.ndarray, n_starts: int = 100,
                              n_steps: int = 16, step_size: float | None = None,
                              k_nn: int = 5, seed: int = 0) -> dict:
    """For each of n_starts random starting activations, walk along w in
    fixed steps and record the average distance to the k-NN in X.

    If the probe direction is intrinsic to the manifold of X, distance stays
    bounded (we walk roughly along the data). If extrinsic, distance grows
    linearly (we leave the manifold).
    """
    rng = np.random.default_rng(seed)
    w = w / (np.linalg.norm(w) + 1e-12)
    if step_size is None:
        # Choose step ~ 1/n_steps of the projected score range (so the walk
        # spans the natural range of the probe in X).
        s = X @ w
        step_size = (s.max() - s.min()) / n_steps
    starts = rng.choice(X.shape[0], size=min(n_starts, X.shape[0]), replace=False)
    nn = NearestNeighbors(n_neighbors=k_nn).fit(X)
    # baseline: typical kNN dist for points in X
    base_d, _ = nn.kneighbors(X[rng.choice(X.shape[0], size=min(200, X.shape[0]), replace=False)])
    baseline = float(np.median(base_d[:, -1]))

    dists = np.zeros((len(starts), n_steps + 1))
    for i, idx in enumerate(starts):
        h = X[idx].copy()
        for t in range(n_steps + 1):
            d, _ = nn.kneighbors(h[None, :])
            dists[i, t] = float(d[0, -1])  # k-th nn distance
            h = h + step_size * w
    mean_dist = dists.mean(axis=0)
    growth = float((mean_dist[-1] - mean_dist[0]) / (baseline + 1e-12))
    return {
        "step_size": float(step_size),
        "baseline_knn_dist": baseline,
        "mean_dist_at_start": float(mean_dist[0]),
        "mean_dist_at_end": float(mean_dist[-1]),
        "growth_in_baselines": growth,  # multiples of typical kNN distance
        "trajectory": mean_dist,
    }


# -- 5. Probe-nullspace nonlinear-dependence R² ----------------------------- #


def nullspace_nonlinear_R2(X: np.ndarray, w: np.ndarray, n_folds: int = 5,
                           n_subsample: int = 1500, seed: int = 0,
                           use_pca_for_X_perp: int = 32) -> dict:
    """Cross-validated R² of predicting s = w·h from the orthogonal projection.

      h_perp = (I - w w^T) h
    Then fit kernel ridge with RBF on h_perp -> s. R²>>0 means activations
    satisfy a nonlinear constraint coupling w·h with other coordinates ⇒ probe
    lies on a manifold of the rest of the activations.

    For a linear-feature direction in Gaussian data, R² → 0.
    """
    rng = np.random.default_rng(seed)
    w = w.astype(np.float64); w = w / (np.linalg.norm(w) + 1e-12)
    s = X @ w
    H = X - s[:, None] * w[None, :]  # h_perp in the original frame
    # Reduce h_perp via PCA so kernel ridge is tractable.
    Hc = H - H.mean(axis=0)
    _, S, Vt = np.linalg.svd(Hc, full_matrices=False)
    k = min(use_pca_for_X_perp, len(S))
    Hp = Hc @ Vt[:k].T  # [N, k]

    if Hp.shape[0] > n_subsample:
        idx = rng.choice(Hp.shape[0], size=n_subsample, replace=False)
        Hp = Hp[idx]; s_sub = s[idx]
    else:
        s_sub = s

    Hp = (Hp - Hp.mean(axis=0)) / (Hp.std(axis=0) + 1e-9)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    sse, sst = 0.0, 0.0
    sse_lin, sst_lin = 0.0, 0.0
    for tr, te in kf.split(Hp):
        # Nonlinear baseline
        gamma = 1.0 / max(Hp.shape[1], 1)
        m = KernelRidge(alpha=1.0, kernel="rbf", gamma=gamma).fit(Hp[tr], s_sub[tr])
        pred = m.predict(Hp[te])
        sse += float(((s_sub[te] - pred) ** 2).sum())
        sst += float(((s_sub[te] - s_sub[tr].mean()) ** 2).sum())
        # Linear baseline (ordinary least squares via ridge with very small alpha)
        ml = KernelRidge(alpha=1e-3, kernel="linear").fit(Hp[tr], s_sub[tr])
        pl = ml.predict(Hp[te])
        sse_lin += float(((s_sub[te] - pl) ** 2).sum())
        sst_lin += float(((s_sub[te] - s_sub[tr].mean()) ** 2).sum())
    R2 = 1 - sse / (sst + 1e-12)
    R2_lin = 1 - sse_lin / (sst_lin + 1e-12)
    return {
        "R2_rbf_kernel": float(R2),
        "R2_linear_baseline": float(R2_lin),
        "nonlinear_gain": float(R2 - R2_lin),
        "n_train_per_fold": int(Hp.shape[0] * (n_folds - 1) / n_folds),
    }


# -- Wrapper that runs all five and reports --------------------------------- #


@dataclass
class FullProbeReport:
    name: str
    pm: ProbeManifold
    gauss: dict
    frame: dict
    tube: dict
    walk: dict
    nullspace: dict


def full_probe_analysis(X: np.ndarray, w: np.ndarray, name: str = "",
                        n_bins: int = 12) -> FullProbeReport:
    pm = fit_probe_manifold(X.astype(np.float64), w.astype(np.float64), n_bins=n_bins)
    return FullProbeReport(
        name=name,
        pm=pm,
        gauss=gauss_map_turning(pm),
        frame=moving_frame_rotation(X, w, n_bins=n_bins),
        tube=tube_anisotropy(X, w, n_bins=n_bins),
        walk=counterfactual_smoothwalk(X, w),
        nullspace=nullspace_nonlinear_R2(X, w),
    )


def report_full(rep: FullProbeReport) -> str:
    return (
        f"[{rep.name}]\n"
        f"  curvature_ratio        = {rep.pm.curvature_ratio:.3f}  (arclen/chord)\n"
        f"  gauss-map turning      = {rep.gauss['total_turning_rad']:.3f} rad  "
        f"(per-step max {rep.gauss['max_per_step_rad']:.3f})\n"
        f"  moving-frame rotation  = {rep.frame['total_rotation_rad']:.3f} rad  "
        f"(per-step max {rep.frame['max_per_step_rad']:.3f})\n"
        f"  tube anisotropy mean   = {rep.tube['anisotropy_mean']:.2f}  "
        f"(max {rep.tube['anisotropy_max']:.2f})\n"
        f"  counterfactual walk    = grew by {rep.walk['growth_in_baselines']:+.2f}× kNN baseline\n"
        f"  nullspace-R² (RBF)     = {rep.nullspace['R2_rbf_kernel']:+.3f}  "
        f"(linear: {rep.nullspace['R2_linear_baseline']:+.3f}; "
        f"nonlinear gain {rep.nullspace['nonlinear_gain']:+.3f})"
    )

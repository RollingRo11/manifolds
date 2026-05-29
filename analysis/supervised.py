"""Supervised manifold (paper-style) used ONLY for validation, never for discovery.

Following Manifold Steering: per-concept centroids in PCA space, then a
parametric curve through them. For cyclic concepts (weekdays, months) we use
a periodic cubic spline; the parameter is just the calendar order.

Also returns geodesic distances *along the spline* between concept centroids,
so we can correlate them with the unsupervised diffusion-map distances.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline


@dataclass
class SupervisedManifold:
    centroids: np.ndarray         # [C, d_pca]
    concepts: list[str]           # length C, in cyclic order
    family: str                   # "weekday" or "month"
    spline: CubicSpline           # parametric on [0, C], periodic
    arclen_total: float


CALENDAR_ORDER = {
    "weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    "month": ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"],
}


def fit_supervised(X: np.ndarray, pids: list[int], labels: dict[int, dict],
                   family: str) -> SupervisedManifold:
    order = CALENDAR_ORDER[family]
    centroids = []
    for c in order:
        rows = [i for i, pid in enumerate(pids)
                if labels[pid]["family"] == family and labels[pid]["concept"] == c]
        if not rows:
            raise ValueError(f"no rows for concept {c}")
        centroids.append(X[rows].mean(axis=0))
    centroids = np.stack(centroids, axis=0)  # [C, d]

    C = len(order)
    t = np.arange(C + 1)  # 0..C, last point repeats first for periodic spline
    Y = np.vstack([centroids, centroids[:1]])
    spline = CubicSpline(t, Y, bc_type="periodic", axis=0)

    # Arc length via fine sampling
    fine = np.linspace(0, C, C * 200)
    pts = spline(fine)
    diffs = np.diff(pts, axis=0)
    arclen_total = float(np.linalg.norm(diffs, axis=1).sum())

    return SupervisedManifold(
        centroids=centroids,
        concepts=order,
        family=family,
        spline=spline,
        arclen_total=arclen_total,
    )


def centroid_geodesic_matrix(sm: SupervisedManifold) -> np.ndarray:
    """Pairwise geodesic distances between concept centroids along the spline."""
    C = len(sm.concepts)
    fine = np.linspace(0, C, C * 200)
    pts = sm.spline(fine)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    # centroid i is at parameter t=i (which is fine index i*200)
    pos = np.array([cum[i * 200] for i in range(C)])
    pos = np.append(pos, cum[-1])  # length total at end
    D = np.zeros((C, C))
    for i in range(C):
        for j in range(C):
            forward = abs(pos[i] - pos[j])
            backward = sm.arclen_total - forward
            D[i, j] = min(forward, backward)
    return D

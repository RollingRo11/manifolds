"""Synthetic test for probe-driven manifold construction.

Build two cases:
  A) Linear feature: data lies in R^d with a true direction t; w == t. Curvature
     ratio should be ≈1 (LRH holds).
  B) Curved feature: data lies on an arc; the probe direction `w` is the chord
     between arc endpoints. Curvature ratio should be > 1 and the local
     tangent should rotate as we walk along the curve.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from probe_manifold import fit_probe_manifold, report


def case_linear(N=400, d=32, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.standard_normal(d); t /= np.linalg.norm(t)
    s = rng.uniform(-1, 1, size=N)
    X = s[:, None] * t[None, :] + noise * rng.standard_normal((N, d))
    return X.astype(np.float32), t.astype(np.float32)


def case_curved(N=400, d=32, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    e1 = rng.standard_normal(d); e1 /= np.linalg.norm(e1)
    e2 = rng.standard_normal(d); e2 -= (e2 @ e1) * e1; e2 /= np.linalg.norm(e2)
    theta = rng.uniform(-np.pi/2, np.pi/2, size=N)
    X = np.cos(theta)[:, None] * e1 + np.sin(theta)[:, None] * e2
    X += noise * rng.standard_normal((N, d))
    # Probe = chord between the two extreme points (theta = -pi/2 and pi/2)
    w = e2.copy()  # chord on unit semicircle: from -e2 to +e2 -> direction e2
    return X.astype(np.float32), w.astype(np.float32)


def main():
    print("--- Case A: linear feature, probe = true direction ---")
    X, w = case_linear()
    pm = fit_probe_manifold(X, w, n_bins=12)
    print(report(pm, "linear"))

    print("\n--- Case B: curved (arc), probe = chord ---")
    X, w = case_curved()
    pm = fit_probe_manifold(X, w, n_bins=12)
    print(report(pm, "curved"))

    # On the unit semicircle from -e2 to +e2, true arclen = pi, chord = 2,
    # so curvature_ratio_target ~ pi/2 = 1.571.
    print(f"  (theoretical curvature ratio for unit semicircle: pi/2 = {np.pi/2:.3f})")


if __name__ == "__main__":
    main()

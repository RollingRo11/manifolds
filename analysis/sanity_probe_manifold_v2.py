"""Sanity tests for the new detectors against synthetic ground-truth cases.

Cases:
  L) Linear feature: data = s*t + noise. All detectors should report ~0
     (or, for kNN walk, no growth).
  C) Curved feature: data on a 2D arc; probe is the chord. Curvature/turning/
     frame-rotation should all be substantially >0.
  T) Heteroscedastic feature: linear in mean but tube width grows with s.
     Curvature/turning are 0; tube_anisotropy.lam1_relvar is large.
  N) Pure nonlinear: data = (s, s^2, ...) + noise. Linear-projection R²
     low, nonlinear nullspace R² high.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from probe_manifold_v2 import full_probe_analysis, report_full


def case_linear(N=600, d=24, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.standard_normal(d); t /= np.linalg.norm(t)
    s = rng.uniform(-1, 1, size=N)
    X = s[:, None] * t[None, :] + noise * rng.standard_normal((N, d))
    return X.astype(np.float32), t.astype(np.float32)


def case_curved(N=600, d=24, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    e1 = rng.standard_normal(d); e1 /= np.linalg.norm(e1)
    e2 = rng.standard_normal(d); e2 -= (e2 @ e1) * e1; e2 /= np.linalg.norm(e2)
    theta = rng.uniform(-np.pi/2, np.pi/2, size=N)
    X = np.cos(theta)[:, None] * e1 + np.sin(theta)[:, None] * e2
    X += noise * rng.standard_normal((N, d))
    return X.astype(np.float32), e2.astype(np.float32)


def case_heteroscedastic(N=600, d=24, noise=0.05, seed=0):
    """Linear in mean but tube width grows with s."""
    rng = np.random.default_rng(seed)
    t = rng.standard_normal(d); t /= np.linalg.norm(t)
    s = rng.uniform(-1, 1, size=N)
    X = s[:, None] * t[None, :]
    # Add noise whose variance depends on s
    sigma = noise * (1 + 5 * (s + 1) / 2)  # noise grows from 1x to 6x as s goes -1 -> +1
    X += sigma[:, None] * rng.standard_normal((N, d))
    return X.astype(np.float32), t.astype(np.float32)


def case_quadratic_perp(N=600, d=24, noise=0.05, seed=0):
    """w·h is a quadratic function of an orthogonal direction.
    s = u^2 + small_noise, where u = e_perp · h.
    """
    rng = np.random.default_rng(seed)
    e1 = rng.standard_normal(d); e1 /= np.linalg.norm(e1)
    e2 = rng.standard_normal(d); e2 -= (e2 @ e1) * e1; e2 /= np.linalg.norm(e2)
    u = rng.uniform(-1, 1, size=N)
    s = u ** 2 + noise * rng.standard_normal(N)
    X = s[:, None] * e1 + u[:, None] * e2 + noise * rng.standard_normal((N, d))
    return X.astype(np.float32), e1.astype(np.float32)


def main():
    print("--- Case L (linear) ---")
    X, w = case_linear(); print(report_full(full_probe_analysis(X, w, "linear")))
    print("\n--- Case C (curved arc, chord probe) ---")
    X, w = case_curved(); print(report_full(full_probe_analysis(X, w, "curved")))
    print("\n--- Case T (heteroscedastic) ---")
    X, w = case_heteroscedastic(); print(report_full(full_probe_analysis(X, w, "hetero")))
    print("\n--- Case N (pure nonlinear: s = u^2) ---")
    X, w = case_quadratic_perp(); print(report_full(full_probe_analysis(X, w, "u^2")))


if __name__ == "__main__":
    main()

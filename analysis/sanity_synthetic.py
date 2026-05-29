"""Sanity check on synthetic data with known geometry.

Build a 1D circle embedded in R^d with noise. Recover it unsupervised; verify:
  - intrinsic_dim ~ 1
  - β1 = 1 (one persistent loop)
  - tangent direction (along the circle) gets tau ~ 1
  - radial direction (perpendicular) gets tau ~ 0

This is the unit test for the manifold pipeline before we point it at real
LLM activations.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from manifold import (
    diffusion_map,
    estimate_tangents,
    knn_geodesic_distances,
    local_pca_eigengap,
    persistence_diagrams,
    reduce_pca,
    tangent_fraction,
    two_nn_intrinsic_dim,
)


def make_circle(N=300, d=64, noise=0.02, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, size=N)
    base = np.zeros((N, d))
    base[:, 0] = np.cos(theta)
    base[:, 1] = np.sin(theta)
    R = rng.standard_normal((d, d))
    Q, _ = np.linalg.qr(R)
    X = base @ Q + noise * rng.standard_normal((N, d))
    # the embedding: e_along(theta) = -sin theta * Q[0] + cos theta * Q[1]
    # the embedding: e_radial(theta) =  cos theta * Q[0] + sin theta * Q[1]
    return X.astype(np.float32), theta, Q


def main():
    X, theta, Q = make_circle()
    print(f"input X.shape={X.shape}")
    pca = reduce_pca(X, k_pca=16)
    print(f"PCA: top 4 explained = {pca.explained[:4]}")
    Xp = pca.X
    id_two = two_nn_intrinsic_dim(Xp)
    eigs, id_local = local_pca_eigengap(Xp, k=20)
    print(f"intrinsic dim: TwoNN={id_two:.2f}  local-PCA-median={id_local}")
    dm = diffusion_map(Xp, k_nn=20, n_eigs=4)
    print(f"diffusion eigs: {dm.eigvals}")
    coords = dm.coords(n_dims=2)
    # Check that the recovered angle reproduces theta (up to a rotation/reflection)
    rec = np.arctan2(coords[:, 1], coords[:, 0])
    # circular correlation
    cc = abs(np.corrcoef(np.cos(rec - theta), np.ones_like(rec))[0, 0])
    print(f"diffusion-coord vs ground-truth angle |cos correlation|={cc:.3f}  (>0.9 = good)")

    print("--- persistent homology ---")
    res = persistence_diagrams(Xp, max_dim=1)
    h0 = res["dgms"][0]
    h1 = res["dgms"][1]
    n_persistent_h1 = sum(1 for (b, d) in h1 if (d - b) > 0.5 * (h1[:, 1] - h1[:, 0]).max())
    print(f"H0 features: {len(h0)}  H1 (loops): {len(h1)}  persistent-H1: {n_persistent_h1}")
    if len(h1) > 0:
        lifetimes = h1[:, 1] - h1[:, 0]
        print(f"top H1 lifetimes: {np.sort(lifetimes)[::-1][:3]}")

    print("--- tangent estimation ---")
    tf = estimate_tangents(Xp, intrinsic_dim=1, k=20)
    # ground truth tangent at each point
    e_along = -np.sin(theta)[:, None] * Q[0] + np.cos(theta)[:, None] * Q[1]
    e_along_pca = pca.components @ e_along.T  # [k_pca, N]
    e_along_pca = e_along_pca.T  # [N, k_pca]
    e_along_pca /= np.linalg.norm(e_along_pca, axis=1, keepdims=True) + 1e-12
    e_radial = np.cos(theta)[:, None] * Q[0] + np.sin(theta)[:, None] * Q[1]
    e_radial_pca = (pca.components @ e_radial.T).T
    e_radial_pca /= np.linalg.norm(e_radial_pca, axis=1, keepdims=True) + 1e-12

    # For each point, compute tau using its own ground-truth tangent (should be ~1) vs radial (~0)
    tau_along = []
    tau_radial = []
    for i in range(len(Xp)):
        a = tf.tangent_basis[i] @ e_along_pca[i]
        tau_along.append(float((a * a).sum()))
        a = tf.tangent_basis[i] @ e_radial_pca[i]
        tau_radial.append(float((a * a).sum()))
    print(f"tau(along)  mean={np.mean(tau_along):.3f}  median={np.median(tau_along):.3f}  (target ~1)")
    print(f"tau(radial) mean={np.mean(tau_radial):.3f}  median={np.median(tau_radial):.3f}  (target ~0)")

    # Random direction baseline
    rng = np.random.default_rng(42)
    w_rand = rng.standard_normal(Xp.shape[1])
    w_rand /= np.linalg.norm(w_rand)
    tau_rand = tangent_fraction(w_rand, tf)
    print(f"tau(random) mean={tau_rand.mean():.3f}  (target ~ 1/{Xp.shape[1]} = {1/Xp.shape[1]:.3f})")

    print("--- geodesic graph distances ---")
    D = knn_geodesic_distances(Xp, k_nn=10)
    # ground truth: arc length on unit circle = |angle diff| (mod 2pi, take min)
    diff = np.abs(theta[:, None] - theta[None, :])
    diff = np.minimum(diff, 2 * np.pi - diff)
    iu = np.triu_indices(len(Xp), k=1)
    rho = np.corrcoef(D[iu], diff[iu])[0, 1]
    print(f"geodesic-graph vs true arc length correlation r={rho:.3f}  (>0.95 = good)")


if __name__ == "__main__":
    main()

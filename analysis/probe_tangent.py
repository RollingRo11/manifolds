"""Probe-tangent decomposition.

For an existing contrastive probe direction w (in *original* activation space):
  1. Project into the same PCA basis used for manifold discovery.
  2. At each manifold point i, compute tau(i) = ||proj_T(w)||^2 / ||w||^2
     (fraction of w lying in T_{x_i} M).
  3. Summarize: mean tau, distribution, and an angle interpretation.

Interpretation:
  tau ~ 1: probe direction is along the manifold (intrinsic coordinate).
  tau ~ 0: probe direction is transverse to the manifold (off-manifold pushes).
  tau ~ d_int / d_pca: probe is uninformed about M (random direction baseline).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from manifold import PCAFit, TangentField, tangent_fraction


@dataclass
class ProbeReport:
    name: str
    tau: np.ndarray              # [N] tangent fraction at each manifold point
    tau_mean: float
    tau_median: float
    random_baseline: float       # E[tau] under random unit direction = d_int / d_pca
    fraction_explained_by_pca: float
    angle_to_manifold_deg_mean: float


def evaluate_probe(name: str, w_full: np.ndarray, pca: PCAFit, tf: TangentField) -> ProbeReport:
    """Project a probe direction defined in original activation space into PCA
    space, then measure its tangent fraction along the manifold."""
    w_full = w_full.astype(np.float32)
    n0 = float(np.linalg.norm(w_full)) + 1e-12

    # Direction in PCA space (relative to the pca mean is irrelevant for direction):
    w_pca_unnorm = pca.components @ w_full
    n_pca = float(np.linalg.norm(w_pca_unnorm))
    fraction_explained = (n_pca / n0) ** 2  # how much of w lies in PCA subspace

    if n_pca < 1e-8:
        return ProbeReport(name=name, tau=np.zeros(tf.tangent_basis.shape[0], dtype=np.float32),
                           tau_mean=0.0, tau_median=0.0,
                           random_baseline=tf.intrinsic_dim / pca.components.shape[0],
                           fraction_explained_by_pca=fraction_explained,
                           angle_to_manifold_deg_mean=90.0)

    tau = tangent_fraction(w_pca_unnorm, tf)
    angles = np.degrees(np.arccos(np.sqrt(np.clip(tau, 0, 1))))

    return ProbeReport(
        name=name,
        tau=tau,
        tau_mean=float(tau.mean()),
        tau_median=float(np.median(tau)),
        random_baseline=tf.intrinsic_dim / pca.components.shape[0],
        fraction_explained_by_pca=fraction_explained,
        angle_to_manifold_deg_mean=float(angles.mean()),
    )


def report_str(r: ProbeReport) -> str:
    return (f"[probe {r.name}] tau mean={r.tau_mean:.3f}  median={r.tau_median:.3f}  "
            f"random-baseline={r.random_baseline:.3f}  "
            f"angle_to_M={r.angle_to_manifold_deg_mean:.1f}°  "
            f"PCA-coverage={r.fraction_explained_by_pca:.3f}")

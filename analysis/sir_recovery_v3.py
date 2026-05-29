"""SIR recovery v3 — global residual-PCA partner axes.

  s        = (X @ w) / ||w||                           # probe coord
  R        = X - s · ŵ                                  # residual (in w⊥ subspace)
  V        = top-k principal components of R (global)   # partner axes
  embed    = (s, R @ V[0], R @ V[1], ...)

This is the simplest possible "recovery from probe": fix one axis to the
probe direction, take the next k axes as the dominant variance directions
of the orthogonal complement.

Theoretical advantage over v1 per-bin local PCA:

  - When w is INFORMATIVE: w typically aligns partially with one of the
    leading PCs of X.  After projecting it out, the top PCs of R are
    approximately the *next* leading PCs of X (which together with w span
    the manifold).  The result is close to supervised PCA, just with
    one axis pinned to the probe.

  - When w is UNINFORMATIVE: projecting out a random 1-D direction barely
    changes the spectrum of X.  The top PCs of R ≈ top PCs of X.  The
    embedding becomes (random_projection, PC1, PC2) — degenerate.  This
    SHOULD show up in NN-accuracy and Procrustes against supervised top-3.

Compared to v1 per-bin PCA, this avoids per-bin estimation noise (we use
all N samples for one global covariance estimation) at the cost of losing
the rotating-tangent intuition that's natural for closed manifolds.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def global_residual_recover(X: np.ndarray, w: np.ndarray, n_partner: int = 2) -> np.ndarray:
    """Returns [N, 1+n_partner] embedding."""
    X = X.astype(np.float64)
    w = w.astype(np.float64); w = w / (np.linalg.norm(w) + 1e-12)
    s = X @ w
    R = X - s[:, None] * w[None, :]
    pca = PCA(n_components=n_partner).fit(R)
    coords = pca.transform(R)
    return np.column_stack([s, coords])

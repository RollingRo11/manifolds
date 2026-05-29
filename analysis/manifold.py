"""Unsupervised manifold discovery and tangent estimation.

The pipeline (purely unsupervised — no concept labels used):

  1. Standardize + PCA-reduce activations to a chosen dimension k_pca (~64).
  2. Estimate intrinsic dimension via two-NN (Levina-Bickel-style) and the
     local-PCA eigengap.
  3. Compute a kNN graph and run diffusion maps for global intrinsic coords.
  4. Estimate the tangent space at each point via local PCA.
  5. (optional) Persistent homology on the kNN-distance metric to characterize
     topology (β0 = components, β1 = loops). A single persistent β1 = 1 is the
     unsupervised signature of a cyclic concept like weekdays/months.

The supervised baseline (for *validation only*, not part of the discovery
algorithm) is in supervised.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


# --------------------------------------------------------------------------- #
# Step 1. Reduce to k_pca dims (still high enough that we don't lose structure)
# --------------------------------------------------------------------------- #


@dataclass
class PCAFit:
    mean: np.ndarray          # [d]
    components: np.ndarray    # [k_pca, d]
    explained: np.ndarray     # [k_pca]
    X: np.ndarray             # [N, k_pca] reduced data

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) @ self.components.T


def reduce_pca(X: np.ndarray, k_pca: int = 64) -> PCAFit:
    pca = PCA(n_components=min(k_pca, *X.shape))
    Xp = pca.fit_transform(X)
    return PCAFit(
        mean=pca.mean_,
        components=pca.components_,
        explained=pca.explained_variance_ratio_,
        X=Xp.astype(np.float32),
    )


# --------------------------------------------------------------------------- #
# Step 2. Intrinsic dimension estimation
# --------------------------------------------------------------------------- #


def two_nn_intrinsic_dim(X: np.ndarray, drop_tail: float = 0.1) -> float:
    """Facco et al. 2017 — TwoNN. ID estimated from r2/r1 ratio of nearest
    neighbor distances. Drop the largest `drop_tail` fraction (heavy tail
    from boundary points) before fitting."""
    nn = NearestNeighbors(n_neighbors=3).fit(X)
    d, _ = nn.kneighbors(X)
    r1 = d[:, 1] + 1e-12
    r2 = d[:, 2] + 1e-12
    mu = r2 / r1
    mu_sorted = np.sort(mu)
    keep = int(len(mu_sorted) * (1 - drop_tail))
    mu_sorted = mu_sorted[:keep]
    F = (np.arange(1, keep + 1) - 0.5) / len(mu)
    # Linear regression of -log(1-F) against log(mu) through origin -> slope = id
    x = np.log(mu_sorted)
    y = -np.log(1 - F)
    slope = (x * y).sum() / (x * x).sum()
    return float(slope)


def local_pca_eigengap(X: np.ndarray, k: int = 30) -> tuple[np.ndarray, float]:
    """At each point, do PCA on its k-NN neighborhood. Return (per-point
    eigenvalues, median estimated intrinsic dim by largest-eigengap rule)."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nn.kneighbors(X)
    eigs = []
    ids = []
    for i in range(X.shape[0]):
        nbrs = X[idx[i, 1:]]
        nbrs = nbrs - nbrs.mean(axis=0, keepdims=True)
        # cov is k x d; svd on nbrs gives singular values whose squares are eigvals.
        s = np.linalg.svd(nbrs, compute_uv=False)
        lam = (s ** 2) / max(k - 1, 1)
        eigs.append(lam)
        # eigengap: index of largest relative drop in the first min(10, len) eigs
        L = lam[: min(10, len(lam))]
        rel = L[:-1] / (L[1:] + 1e-12)
        d_hat = int(np.argmax(rel)) + 1
        ids.append(d_hat)
    eigs_padded = np.zeros((X.shape[0], max(len(e) for e in eigs)))
    for i, e in enumerate(eigs):
        eigs_padded[i, : len(e)] = e
    return eigs_padded, float(np.median(ids))


# --------------------------------------------------------------------------- #
# Step 3. Diffusion maps for global intrinsic coordinates
# --------------------------------------------------------------------------- #


@dataclass
class DiffusionMap:
    eigvals: np.ndarray   # [k+1] (first = 1 trivial)
    eigvecs: np.ndarray   # [N, k+1]
    sigma: float

    def coords(self, n_dims: int = 3, t: int = 1) -> np.ndarray:
        # Skip the trivial first eigenvector; weight by eigenvalue^t for time t.
        return self.eigvecs[:, 1 : n_dims + 1] * (self.eigvals[1 : n_dims + 1] ** t)


def diffusion_map(
    X: np.ndarray,
    k_nn: int = 15,
    n_eigs: int = 8,
    sigma: float | None = None,
    alpha: float = 1.0,
) -> DiffusionMap:
    """Anisotropic diffusion map (Coifman-Lafon). alpha=1 cancels density bias
    so eigenvectors approximate Laplace-Beltrami eigenfunctions of the manifold."""
    nn = NearestNeighbors(n_neighbors=k_nn + 1).fit(X)
    d, idx = nn.kneighbors(X)
    if sigma is None:
        # Bandwidth = median of k_nn-th neighbor distance
        sigma = float(np.median(d[:, k_nn]))
    N = X.shape[0]
    rows = np.repeat(np.arange(N), k_nn + 1)
    cols = idx.reshape(-1)
    vals = np.exp(-(d.reshape(-1) ** 2) / (2 * sigma ** 2))
    K = csr_matrix((vals, (rows, cols)), shape=(N, N))
    K = (K + K.T) / 2  # symmetrize
    # Anisotropic normalization
    deg = np.asarray(K.sum(axis=1)).ravel()
    Dinv = 1.0 / (deg ** alpha + 1e-12)
    K2 = csr_matrix((K.data * Dinv[K.nonzero()[0]] * Dinv[K.nonzero()[1]],
                     (K.nonzero()[0], K.nonzero()[1])), shape=K.shape)
    deg2 = np.asarray(K2.sum(axis=1)).ravel()
    # Symmetric normalization for eigsh stability
    Dh = 1.0 / np.sqrt(deg2 + 1e-12)
    P_sym = csr_matrix((K2.data * Dh[K2.nonzero()[0]] * Dh[K2.nonzero()[1]],
                        (K2.nonzero()[0], K2.nonzero()[1])), shape=K2.shape)
    vals_, vecs_ = eigsh(P_sym, k=n_eigs + 1, which="LA")
    order = np.argsort(-vals_)
    vals_ = vals_[order]
    vecs_ = vecs_[:, order]
    # Convert back to the asymmetric P's right eigenvectors
    vecs_ = vecs_ * Dh[:, None]
    # Normalize each column
    vecs_ = vecs_ / (np.linalg.norm(vecs_, axis=0, keepdims=True) + 1e-12)
    return DiffusionMap(eigvals=vals_, eigvecs=vecs_, sigma=sigma)


# --------------------------------------------------------------------------- #
# Step 4. Tangent space estimation at each point (local PCA)
# --------------------------------------------------------------------------- #


@dataclass
class TangentField:
    """At each of N points, the d-dim ambient space is decomposed into a
    local tangent subspace of dim `intrinsic_dim` (from local PCA on k-NN)
    and the orthogonal normal subspace.

    tangent_basis[i]: [intrinsic_dim, d] orthonormal rows spanning T_{x_i} M
    eigvals[i]:       full spectrum at point i (decreasing)
    """
    tangent_basis: np.ndarray
    eigvals: np.ndarray
    intrinsic_dim: int
    nn_idx: np.ndarray  # [N, k+1]


def estimate_tangents(X: np.ndarray, intrinsic_dim: int, k: int = 30) -> TangentField:
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nn.kneighbors(X)
    N, d = X.shape
    bases = np.empty((N, intrinsic_dim, d), dtype=np.float32)
    eigs = np.empty((N, min(k, d)), dtype=np.float32)
    for i in range(N):
        nbrs = X[idx[i, 1:]] - X[i]  # center on the point itself for tangent
        # SVD: nbrs = U S Vt; rows of Vt are principal directions in ambient space
        U, S, Vt = np.linalg.svd(nbrs, full_matrices=False)
        eigs[i, : len(S)] = (S ** 2) / max(k - 1, 1)
        bases[i] = Vt[:intrinsic_dim].astype(np.float32)
    return TangentField(tangent_basis=bases, eigvals=eigs,
                        intrinsic_dim=intrinsic_dim, nn_idx=idx)


# --------------------------------------------------------------------------- #
# Tangent fraction of an arbitrary direction (the probe-manifold test)
# --------------------------------------------------------------------------- #


def tangent_fraction(direction: np.ndarray, tf: TangentField) -> np.ndarray:
    """For each point, fraction of `direction`'s squared norm that lies in
    T_{x_i} M. Returns array of shape [N] in [0, 1].

    Decomposition: w = w_par + w_perp with w_par = B^T B w (B rows orthonormal).
    tau(i) = ||w_par||^2 / ||w||^2.
    """
    w = direction.astype(np.float32)
    w = w / (np.linalg.norm(w) + 1e-12)
    # bases[i]: [d_int, d]; project: a = bases[i] @ w; w_par_norm2 = a @ a
    a = tf.tangent_basis @ w  # [N, d_int]
    return (a * a).sum(axis=1)


# --------------------------------------------------------------------------- #
# Geodesic distances on the kNN graph (for cross-manifold comparisons)
# --------------------------------------------------------------------------- #


def knn_geodesic_distances(X: np.ndarray, k_nn: int = 15,
                           auto_extend: bool = True, max_k: int = 80) -> np.ndarray:
    """kNN graph + Dijkstra. If the kNN graph is disconnected, optionally
    increase k until it becomes connected (capped at max_k)."""
    N = X.shape[0]
    k = k_nn
    while True:
        nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
        d, idx = nn.kneighbors(X)
        rows = np.repeat(np.arange(N), k + 1)
        cols = idx.reshape(-1)
        vals = d.reshape(-1)
        G = csr_matrix((vals, (rows, cols)), shape=(N, N))
        G = G.maximum(G.T)
        n_comp, _ = connected_components(G, directed=False)
        if n_comp == 1 or not auto_extend or k >= max_k:
            break
        k = min(max_k, int(k * 1.5))
    if n_comp != 1:
        raise RuntimeError(f"kNN graph disconnected ({n_comp} components) even at k={k}")
    return dijkstra(G, directed=False)


# --------------------------------------------------------------------------- #
# Persistent homology — topology fingerprint
# --------------------------------------------------------------------------- #


def persistence_diagrams(X: np.ndarray, max_dim: int = 1):
    """Vietoris-Rips persistent homology up to dim `max_dim`. Returns the
    ripser dict (dgms[0] = β0 lifetimes, dgms[1] = β1 = loops)."""
    from ripser import ripser
    return ripser(X, maxdim=max_dim)

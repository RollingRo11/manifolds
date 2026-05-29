"""SAVE with 2 bins = aware vs non-aware on FORTRESS.

Standard SAVE uses K probe-quantile slices and finds directions where
within-slice covariance differs from total. With K=2 bins set EXACTLY to the
class labels, we ask: 'in what directions do aware and non-aware activations
have DIFFERENT covariance structure?' This is orthogonal-or-different to the
mean-difference direction (w_fortress).

If the top SAVE_2bin eigenvector ≠ w_fortress, that's a new direction that
captures higher-order class structure. Could be a steering direction.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))
from load_acts_multi import read_last_token_multi_rollout

LAYER = 40
FORTRESS_SHARD = Path("/data/artifacts/rohan/activation-harvester/fortress_balanced_v1/shard_dp00")
FORTRESS_LABELS = Path("/home/rkathuria/santi/data/probe_train/rollout_labels_fortress_balanced.jsonl")
OUT_DIR = Path("/home/rkathuria/manifolds/figures/v11_save_2bin")


def normalize(v):
    return v / (np.linalg.norm(v) + 1e-12)


def save_with_explicit_bins(X: np.ndarray, bin_labels: np.ndarray, d_M: int = 5):
    """SAVE with arbitrary discrete bin labels (not probe-quantile-based).

    M_SAVE = (1/N) Σ_b n_b (Σ - Σ_b)²
    where Σ is total cov, Σ_b is within-bin cov.
    Returns (eigvals, eigvecs) sorted descending.
    """
    X = X.astype(np.float64)
    Xc = X - X.mean(0, keepdims=True)
    N, d = X.shape

    # Thin SVD basis (work in K-dim, K = rank of Xc)
    U, S, Qt = np.linalg.svd(Xc, full_matrices=False)
    keep = S > S.max() * 1e-10
    Q = Qt[keep].T          # [d, K]
    A = Xc @ Q              # [N, K]

    Sigma_A = (A.T @ A) / N

    M = np.zeros((Q.shape[1], Q.shape[1]))
    bins = np.unique(bin_labels)
    for b in bins:
        mask = bin_labels == b
        nb = int(mask.sum())
        if nb < 2:
            continue
        Ab = A[mask] - A[mask].mean(0, keepdims=True)
        Sigma_b = (Ab.T @ Ab) / nb
        diff = Sigma_A - Sigma_b
        M += (nb / N) * (diff @ diff)

    eigvals, eigvecs = np.linalg.eigh(M)
    order = np.argsort(-eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Lift back to ambient
    save_dirs = Q @ eigvecs[:, :d_M]
    save_dirs /= (np.linalg.norm(save_dirs, axis=0, keepdims=True) + 1e-12)

    return eigvals, save_dirs


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("loading FORTRESS L40 multi-rollout...")
    X, keys = read_last_token_multi_rollout(FORTRESS_SHARD, LAYER, position_from_end=0)
    aware_map = {}
    with open(FORTRESS_LABELS) as f:
        for line in f:
            d = json.loads(line)
            aware_map[(int(d["prompt_id"]), int(d["completion_idx"]))] = bool(d["aware"])
    matched = [(i, aware_map[k]) for i, k in enumerate(keys) if k in aware_map]
    idx = np.array([m[0] for m in matched])
    aware = np.array([m[1] for m in matched])
    Xa = X[idx].astype(np.float64)
    print(f"  matched {Xa.shape[0]} acts, aware={aware.sum()} non={(~aware).sum()}")

    # Mean difference direction (for comparison)
    mean_a = Xa[aware].mean(0); mean_n = Xa[~aware].mean(0)
    w_fortress = normalize(mean_a - mean_n)
    print(f"  ‖mean_a − mean_n‖ = {np.linalg.norm(mean_a - mean_n):.2f}")

    # ===== SAVE with 2 bins = aware/non-aware =====
    print("\n=== SAVE with 2 bins (aware vs non-aware) ===")
    bin_labels = aware.astype(int)
    eigvals, save_dirs = save_with_explicit_bins(Xa, bin_labels, d_M=10)

    print(f"top-10 eigenvalues:")
    for i, ev in enumerate(eigvals[:10]):
        print(f"  λ[{i}] = {ev:.2f}")
    print(f"  λ[0] / λ[20] = {eigvals[0] / max(eigvals[20], 1e-12):.2f}")

    # Compare top SAVE direction to w_fortress
    print("\n=== alignment of SAVE_2bin eigvecs with w_fortress ===")
    for i in range(min(10, save_dirs.shape[1])):
        c = save_dirs[:, i] @ w_fortress
        print(f"  cos(SAVE_dir{i}, w_fortress) = {c:+.4f}")

    # ===== for each top SAVE dir: how does it separate the classes? =====
    print("\n=== class-discrimination quality of top SAVE_2bin directions ===")
    print(f"{'dir':>5}  {'mean_a':>8}  {'mean_n':>8}  {'std_a':>7}  {'std_n':>7}  {'diff/sqrt(σ²+σ²)':>17}  {'cos(w_fortress)':>17}")
    for i in range(min(10, save_dirs.shape[1])):
        v = save_dirs[:, i]
        sa = Xa[aware] @ v
        sn = Xa[~aware] @ v
        dscore = (sa.mean() - sn.mean()) / np.sqrt(sa.var() + sn.var() + 1e-12)
        c = v @ w_fortress
        print(f"  {i:>5}  {sa.mean():+8.3f}  {sn.mean():+8.3f}  "
              f"{sa.std():7.3f}  {sn.std():7.3f}  {dscore:+17.4f}  {c:+17.4f}")

    # Save
    np.savez(OUT_DIR / "save_2bin.npz",
              eigvals=eigvals, save_dirs=save_dirs, w_fortress=w_fortress)
    print(f"\nwrote {OUT_DIR}/save_2bin.npz")


if __name__ == "__main__":
    main()

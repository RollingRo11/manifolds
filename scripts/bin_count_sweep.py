"""Which estimator robustly recovers the number of concept bins (clusters)?

For days (true K=7) and months (true K=12), sweep layers and compare K-estimators:
  - spectral_eigengap : eigengap of the normalized graph Laplacian (the principled
                        "eigenvalue" bin counter used by spectral clustering)
  - silhouette        : argmax silhouette over KMeans
  - GMM-BIC           : argmin BIC over Gaussian mixtures
  - calinski_harabasz : argmax CH score over KMeans
  - save_eigratio     : largest SAVE-matrix λ[i]/λ[i+1] drop (the original heuristic)

Writes a JSON table of estimates per (family, layer, method) plus ARI@trueK, so
we can see which method lands on the true K consistently. Designed to run as a
slurm CPU job (see submit_bin_count_sweep.sh), not on the login node.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))
from load_acts import load_labels, read_last_token_per_seq
from save_recovery import save_recovery

SHARD = Path("/data/artifacts/rohan/manifolds/concepts_olmo31_32b_v3/shard_dp00")
LABELS = Path("/home/rkathuria/manifolds/data/labels_v3.jsonl")
OUT = Path("/home/rkathuria/manifolds/figures/bin_count_sweep")
LAYERS = [16, 24, 32, 40, 48, 56]
FAMILIES = {
    "weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                "Saturday", "Sunday"],
    "month": ["January", "February", "March", "April", "May", "June", "July",
              "August", "September", "October", "November", "December"],
}


def normalize(v):
    return v / (np.linalg.norm(v) + 1e-12)


def spectral_eigengap(Xp, k_nn, kmax):
    A = kneighbors_graph(Xp, k_nn, mode="connectivity", include_self=False)
    A = 0.5 * (A + A.T)
    d = np.asarray(A.sum(1)).ravel(); d[d == 0] = 1
    Dinv = 1 / np.sqrt(d)
    L = np.eye(A.shape[0]) - (Dinv[:, None] * A.toarray() * Dinv[None, :])
    ev = np.sort(np.linalg.eigvalsh(L))[:kmax + 1]
    gaps = np.diff(ev)
    return int(np.argmax(gaps[:kmax])) + 1, ev.tolist()


def silhouette_best(Xp, kmax):
    scores = {k: silhouette_score(Xp, KMeans(k, n_init=5, random_state=0).fit_predict(Xp))
              for k in range(2, kmax + 1)}
    return int(max(scores, key=scores.get)), scores


def ch_best(Xp, kmax):
    scores = {k: calinski_harabasz_score(Xp, KMeans(k, n_init=5, random_state=0).fit_predict(Xp))
              for k in range(2, kmax + 1)}
    return int(max(scores, key=scores.get))


def bic_best(Xp, kmax):
    bics = {k: GaussianMixture(k, covariance_type="diag", random_state=0).fit(Xp).bic(Xp)
            for k in range(2, kmax + 1)}
    return int(min(bics, key=bics.get))


def save_eigratio(Xf, ci, K, kmax):
    cents = np.stack([Xf[ci == k].mean(0) for k in range(K)])
    res = save_recovery(Xf, normalize(cents[-1] - cents[0]), n_slices=10, d_M=3)
    e = res.eigvals
    k = min(kmax, len(e) - 1)
    ratios = [e[i] / max(e[i + 1], 1e-12) for i in range(k)]
    return int(np.argmax(ratios)) + 1


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labels = load_labels(LABELS)
    results = []
    for fam, order in FAMILIES.items():
        idx = {c: i for i, c in enumerate(order)}
        K = len(order)
        kmax = 2 * K
        for L in LAYERS:
            X, pids = read_last_token_per_seq(SHARD, L)
            keep = [i for i, p in enumerate(pids) if labels[p].get("family") == fam]
            Xf = X[keep].astype(np.float64)
            ci = np.array([idx[labels[pids[i]]["concept"]] for i in keep])
            Xp = PCA(min(50, Xf.shape[0], Xf.shape[1]), random_state=0).fit_transform(
                Xf - Xf.mean(0))
            eg, ev = spectral_eigengap(Xp, k_nn=15, kmax=kmax + 3)
            sb, sil = silhouette_best(Xp, kmax)
            row = {
                "family": fam, "layer": L, "true_K": K, "N": int(Xf.shape[0]),
                "spectral_eigengap": eg, "silhouette": sb,
                "calinski_harabasz": ch_best(Xp, kmax), "GMM_BIC": bic_best(Xp, kmax),
                "save_eigratio": save_eigratio(Xf, ci, K, kmax),
                "ARI_at_trueK": float(adjusted_rand_score(
                    ci, KMeans(K, n_init=5, random_state=0).fit_predict(Xp))),
                "laplacian_eigs": ev,
                "silhouette_curve": sil,
            }
            results.append(row)
            print(f"{fam:8} L{L:02d} (K={K}): eigengap={eg:2d} sil={sb:2d} "
                  f"CH={row['calinski_harabasz']:2d} BIC={row['GMM_BIC']:2d} "
                  f"save_ratio={row['save_eigratio']:2d}  ARI@K={row['ARI_at_trueK']:.2f}",
                  flush=True)

    (OUT / "bin_count_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}/bin_count_results.json")

    # Summary: how often each method hits the true K (exactly / within ±1).
    methods = ["spectral_eigengap", "silhouette", "calinski_harabasz",
               "GMM_BIC", "save_eigratio"]
    print("\n=== hit rate (exact / within +-1 of true K) across all family x layer ===")
    for m in methods:
        exact = sum(r[m] == r["true_K"] for r in results)
        near = sum(abs(r[m] - r["true_K"]) <= 1 for r in results)
        print(f"  {m:18}: exact {exact}/{len(results)}   within±1 {near}/{len(results)}")


if __name__ == "__main__":
    main()

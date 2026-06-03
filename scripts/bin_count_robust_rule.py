"""Sturdier K-selection so DAYS recovers 7 at L40 too (not just months=12).

The Laplacian eigengap counts graph components; at L40 the days sit on a
connected curved manifold (KMeans-separable, ARI=1.0, but not disconnected), so
the eigengap reads 8. Clustering-QUALITY selectors don't need disconnection.
We test, on the K-agnostic cleaned representation (remove top-1 global PC):

  silhouette        : argmax silhouette (KMeans)
  calinski_harabasz : argmax CH
  davies_bouldin    : argmin DB
  gap_statistic     : Tibshirani gap vs uniform null (smallest K past the knee)
  prediction_strength: largest K with PS >= 0.8 (Tibshirani 2005)

Reports recovered K per (family, layer, method); saves L40 per-K curves for
plotting. K is never supplied. Run as a slurm CPU job.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             pairwise_distances_argmin, silhouette_score)

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))
from load_acts import load_labels, read_last_token_per_seq

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
RNG = np.random.RandomState(0)


def rm_top1_pca(Xf, k=10):
    """Remove top global PC, keep a modest-dim PCA (clustering-quality rules are
    happier in low dim than in 50-D)."""
    Xc = Xf - Xf.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Xrm = Xc - (Xc @ Vt[:1].T) @ Vt[:1]
    return PCA(min(k, *Xrm.shape), random_state=0).fit_transform(Xrm)


def km(X, k):
    return KMeans(k, n_init=10, random_state=0).fit(X)


def Wk(X, lab, k):
    w = 0.0
    for c in range(k):
        pts = X[lab == c]
        if len(pts) > 1:
            w += ((pts - pts.mean(0)) ** 2).sum()
    return w


def gap_statistic(X, Krange, B=10):
    mins, maxs = X.min(0), X.max(0)
    logW = {}; logWstar = {}; sk = {}
    for k in Krange:
        logW[k] = np.log(Wk(X, km(X, k).labels_, k) + 1e-12)
        refs = []
        for _ in range(B):
            Xr = RNG.uniform(mins, maxs, size=X.shape)
            refs.append(np.log(Wk(Xr, km(Xr, k).labels_, k) + 1e-12))
        logWstar[k] = np.mean(refs)
        sk[k] = np.std(refs) * np.sqrt(1 + 1.0 / B)
    gap = {k: logWstar[k] - logW[k] for k in Krange}
    for k in Krange[:-1]:
        if gap[k] >= gap[k + 1] - sk[k + 1]:
            return k, gap
    return Krange[-1], gap


def prediction_strength(X, Krange, n_rep=5, thr=0.8):
    n = X.shape[0]; ps_curve = {}
    for k in Krange:
        scores = []
        for _ in range(n_rep):
            perm = RNG.permutation(n); half = n // 2
            tr, te = X[perm[:half]], X[perm[half:]]
            ctr = km(tr, k).cluster_centers_
            te_lab = km(te, k).labels_
            te_to_tr = pairwise_distances_argmin(te, ctr)  # assign test by train centroids
            worst = 1.0
            for c in range(k):
                idx = np.where(te_lab == c)[0]
                if len(idx) <= 1:
                    continue
                same = te_to_tr[idx][:, None] == te_to_tr[idx][None, :]
                np.fill_diagonal(same, False)
                worst = min(worst, same.sum() / (len(idx) * (len(idx) - 1)))
            scores.append(worst)
        ps_curve[k] = float(np.mean(scores))
    best = max([k for k in Krange if ps_curve[k] >= thr], default=Krange[0])
    return best, ps_curve


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labels = load_labels(LABELS)
    results = []; l40_curves = {}
    for fam, order in FAMILIES.items():
        idx = {c: i for i, c in enumerate(order)}
        K = len(order); Krange = list(range(2, 2 * K + 1))
        for L in LAYERS:
            X, pids = read_last_token_per_seq(SHARD, L)
            keep = [i for i, p in enumerate(pids) if labels[p].get("family") == fam]
            Xf = X[keep].astype(np.float64)
            Xp = rm_top1_pca(Xf)
            sil = {k: silhouette_score(Xp, km(Xp, k).labels_) for k in Krange}
            ch = {k: calinski_harabasz_score(Xp, km(Xp, k).labels_) for k in Krange}
            db = {k: davies_bouldin_score(Xp, km(Xp, k).labels_) for k in Krange}
            gap_k, gap = gap_statistic(Xp, Krange)
            ps_k, ps = prediction_strength(Xp, Krange)
            rec = {"silhouette": max(sil, key=sil.get),
                   "calinski_harabasz": max(ch, key=ch.get),
                   "davies_bouldin": min(db, key=db.get),
                   "gap_statistic": gap_k, "prediction_strength": ps_k}
            results.append({"family": fam, "layer": L, "true_K": K, **rec})
            if L == 40:
                l40_curves[fam] = {"silhouette": sil, "gap": gap,
                                   "prediction_strength": ps, "true_K": K}
            print(f"{fam:8} L{L:02d} K={K}: " +
                  " ".join(f"{m}={v}" for m, v in rec.items()), flush=True)

    (OUT / "bin_count_rule_results.json").write_text(json.dumps(results, indent=2))
    (OUT / "bin_count_rule_L40_curves.json").write_text(json.dumps(l40_curves, indent=2))

    methods = ["silhouette", "calinski_harabasz", "davies_bouldin",
               "gap_statistic", "prediction_strength"]
    print("\n=== exact-K hits [max 12] and L40 (max 2) ===")
    for m in methods:
        exact = sum(r[m] == r["true_K"] for r in results)
        l40 = sum(r[m] == r["true_K"] for r in results if r["layer"] == 40)
        print(f"  {m:20}: exact {exact:2d}/12  | L40 {l40}/2")
    print(f"\nwrote {OUT}/bin_count_rule_results.json + _L40_curves.json")


if __name__ == "__main__":
    main()

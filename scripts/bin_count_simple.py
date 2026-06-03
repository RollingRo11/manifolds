"""Simplest K-selection that works at EVERY layer: silhouette vs eigengap on
simple normalizations (unit-norm / cosine space, remove-top-PC), all layers,
days + months. Goal: a one-liner rule whose recovered K = true K at all 6 layers.

Outputs a recovered-K-vs-layer table (saved to JSON) so "works everywhere" is
verifiable at a glance. K is never supplied. Run as a slurm CPU job.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import kneighbors_graph

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


def reps(Xf, k=20):
    Xc = Xf - Xf.mean(0)
    Xn = Xf / (np.linalg.norm(Xf, axis=1, keepdims=True) + 1e-12)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Xrm = Xc - (Xc @ Vt[:1].T) @ Vt[:1]
    # cosine space + remove top PC: normalize, then drop the shared direction
    Xnc = Xn - Xn.mean(0)
    _, _, Vtn = np.linalg.svd(Xnc, full_matrices=False)
    Xnrm = Xnc - (Xnc @ Vtn[:1].T) @ Vtn[:1]
    P = lambda Z: PCA(min(k, *Z.shape), random_state=0).fit_transform(Z)
    return {"unitnorm": P(Xn - Xn.mean(0)), "rm_top1": P(Xrm),
            "unitnorm_rm_top1": P(Xnrm)}


def sil_K(Xp, Krange):
    s = {k: silhouette_score(Xp, KMeans(k, n_init=10, random_state=0).fit_predict(Xp))
         for k in Krange}
    return int(max(s, key=s.get))


def eigengap_K(Xp, kmax, k_nn=15):
    A = kneighbors_graph(Xp, k_nn, mode="connectivity", include_self=False)
    A = 0.5 * (A + A.T)
    d = np.asarray(A.sum(1)).ravel(); d[d == 0] = 1
    Dinv = 1 / np.sqrt(d)
    L = np.eye(A.shape[0]) - (Dinv[:, None] * A.toarray() * Dinv[None, :])
    ev = np.sort(np.linalg.eigvalsh(L))[:kmax + 1]
    return int(np.argmax(np.diff(ev)[:kmax])) + 1


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labels = load_labels(LABELS)
    results = []
    for fam, order in FAMILIES.items():
        K = len(order); Krange = list(range(2, 2 * K + 1))
        for L in LAYERS:
            X, pids = read_last_token_per_seq(SHARD, L)
            keep = [i for i, p in enumerate(pids) if labels[p].get("family") == fam]
            Xf = X[keep].astype(np.float64)
            row = {"family": fam, "layer": L, "true_K": K}
            for rname, Xp in reps(Xf).items():
                row[f"sil_{rname}"] = sil_K(Xp, Krange)
                row[f"eig_{rname}"] = eigengap_K(Xp, 2 * K)
            results.append(row)
            print(f"{fam:8} L{L:02d} K={K}: " +
                  " ".join(f"{k}={v}" for k, v in row.items()
                           if k not in ("family", "layer", "true_K")), flush=True)

    (OUT / "bin_count_simple_results.json").write_text(json.dumps(results, indent=2))
    combos = [c for c in results[0] if c.startswith(("sil_", "eig_"))]
    print("\n=== exact-K hits [max 12] | days-all-6 | months-all-6 ===")
    for c in combos:
        ex = sum(r[c] == r["true_K"] for r in results)
        dall = all(r[c] == 7 for r in results if r["family"] == "weekday")
        mall = all(r[c] == 12 for r in results if r["family"] == "month")
        print(f"  {c:22}: exact {ex:2d}/12  days6 {dall}  months6 {mall}")
    print(f"\nwrote {OUT}/bin_count_simple_results.json")


if __name__ == "__main__":
    main()

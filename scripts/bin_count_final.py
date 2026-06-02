"""Lock down a layer-agnostic bin-count rule that works at L40 (the viz layer).

From the representation sweep: removing the top global PCs makes the concept
clusters perfectly separable at every layer (ARI@K -> 1.0 at L40). The remaining
question is the K-selection rule. The Laplacian eigengap with a kNN graph can
lock onto coarse sub-structure (e.g. weekday/weekend) when neighbors bridge
adjacent clusters, so we sweep k_nn on the cleaned representation and find the
setting that recovers true K across layers.

For rep in {rm_top1, rm_top2} and k_nn in a grid, report eigengap-recovered K
per (family, layer); save the L40 Laplacian spectra for plotting. K is never
supplied. Run as a slurm CPU job.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))
from load_acts import load_labels, read_last_token_per_seq

SHARD = Path("/data/artifacts/rohan/manifolds/concepts_olmo31_32b_v3/shard_dp00")
LABELS = Path("/home/rkathuria/manifolds/data/labels_v3.jsonl")
OUT = Path("/home/rkathuria/manifolds/figures/bin_count_sweep")
LAYERS = [16, 24, 32, 40, 48, 56]
K_NN_GRID = [6, 8, 10, 12, 15, 20, 30]
REPS = [1, 2]   # remove top-q global PCs
FAMILIES = {
    "weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                "Saturday", "Sunday"],
    "month": ["January", "February", "March", "April", "May", "June", "July",
              "August", "September", "October", "November", "December"],
}


def rm_top_pca(Xf, q, k=50):
    Xc = Xf - Xf.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Xrm = Xc - (Xc @ Vt[:q].T) @ Vt[:q]
    return PCA(min(k, *Xrm.shape), random_state=0).fit_transform(Xrm)


def laplacian_eigs(Xp, k_nn, kmax):
    A = kneighbors_graph(Xp, k_nn, mode="connectivity", include_self=False)
    A = 0.5 * (A + A.T)
    d = np.asarray(A.sum(1)).ravel(); d[d == 0] = 1
    Dinv = 1 / np.sqrt(d)
    L = np.eye(A.shape[0]) - (Dinv[:, None] * A.toarray() * Dinv[None, :])
    return np.sort(np.linalg.eigvalsh(L))[:kmax + 1]


def eigengap_K(ev, kmax):
    return int(np.argmax(np.diff(ev)[:kmax])) + 1


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labels = load_labels(LABELS)
    grid, spectra = [], {}
    for fam, order in FAMILIES.items():
        idx = {c: i for i, c in enumerate(order)}
        K = len(order); kmax = 2 * K
        for L in LAYERS:
            X, pids = read_last_token_per_seq(SHARD, L)
            keep = [i for i, p in enumerate(pids) if labels[p].get("family") == fam]
            Xf = X[keep].astype(np.float64)
            for q in REPS:
                Xp = rm_top_pca(Xf, q)
                for knn in K_NN_GRID:
                    ev = laplacian_eigs(Xp, knn, kmax + 3)
                    grid.append({"family": fam, "layer": L, "true_K": K,
                                 "rm_top": q, "k_nn": knn,
                                 "eigengap_K": eigengap_K(ev, kmax)})
                    if L == 40:
                        spectra[f"{fam}_rmtop{q}_knn{knn}"] = ev.tolist()
            best = {knn: next(g["eigengap_K"] for g in grid
                              if g["family"] == fam and g["layer"] == L
                              and g["rm_top"] == 2 and g["k_nn"] == knn)
                    for knn in K_NN_GRID}
            print(f"{fam:8} L{L:02d} K={K} rm_top2 eigengap by k_nn: {best}", flush=True)

    (OUT / "bin_count_final_grid.json").write_text(json.dumps(grid, indent=2))
    (OUT / "bin_count_final_L40_spectra.json").write_text(json.dumps(spectra, indent=2))

    # Which (rm_top, k_nn) recovers true K on the most (family,layer) cells?
    print("\n=== layer-agnostic exact-K hits per (rm_top, k_nn)  [max 12] ===")
    best_cfg, best_hits = None, -1
    for q in REPS:
        for knn in K_NN_GRID:
            cells = [g for g in grid if g["rm_top"] == q and g["k_nn"] == knn]
            exact = sum(g["eigengap_K"] == g["true_K"] for g in cells)
            l40 = sum(g["eigengap_K"] == g["true_K"] for g in cells if g["layer"] == 40)
            print(f"  rm_top{q} k_nn={knn:2d}: exact {exact:2d}/12  | L40 {l40}/2")
            if (exact, l40) > (best_hits, 0):
                best_hits, best_cfg = exact, (q, knn)
    print(f"\nmost layer-agnostic config: rm_top{best_cfg[0]}, k_nn={best_cfg[1]} "
          f"({best_hits}/12 exact)")
    print(f"wrote {OUT}/bin_count_final_grid.json and _L40_spectra.json")


if __name__ == "__main__":
    main()

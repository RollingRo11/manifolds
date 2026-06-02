"""Layer-agnostic bin-count recovery: which representation makes the Laplacian
eigengap recover the true K at EVERY layer (especially L40, the steering /
visualization layer)?

Raw deep-layer activations smear concept clusters (context variance + transformer
"massive activations" dominate). We test K-agnostic preprocessings that remove
that nuisance variance, then read K off the graph-Laplacian eigengap:

  raw_pca50      : baseline (PCA-50 of centered acts)
  unitnorm       : L2-normalize each row first (kills magnitude / massive acts)
  zscore         : standardize each dim first
  rm_top{1,2,3}  : project out the top global PCs (shared context directions)
  save_embed     : project onto SAVE central subspace (probe = c_last-c_first;
                   uses 2 anchor concepts, NOT the count K)

None of these are told K. Reports recovered-K and ARI@trueK per
(family, layer, representation), plus which representation is most layer-agnostic.
Run as a slurm CPU job (submit_bin_count_repr_sweep.sh), not on the login node.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
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
K_NN = 15


def normalize(v):
    return v / (np.linalg.norm(v) + 1e-12)


def pca(X, k=50):
    return PCA(min(k, *X.shape), random_state=0).fit_transform(X - X.mean(0))


def representations(Xf, cents):
    """K-agnostic preprocessings -> low-dim coords for graph construction."""
    Xc = Xf - Xf.mean(0)
    reps = {"raw_pca50": pca(Xc)}
    Xn = Xf / (np.linalg.norm(Xf, axis=1, keepdims=True) + 1e-12)
    reps["unitnorm"] = pca(Xn)
    reps["zscore"] = pca((Xf - Xf.mean(0)) / (Xf.std(0) + 1e-8))
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    for q in (1, 2, 3):
        Xrm = Xc - (Xc @ Vt[:q].T) @ Vt[:q]
        reps[f"rm_top{q}"] = pca(Xrm)
    # SAVE central subspace (probe from 2 anchor concepts, not the count)
    res = save_recovery(Xf, normalize(cents[-1] - cents[0]), n_slices=10, d_M=20)
    reps["save_embed"] = res.embedding
    return reps


def eigengap_K(Xp, kmax):
    A = kneighbors_graph(Xp, K_NN, mode="connectivity", include_self=False)
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
        idx = {c: i for i, c in enumerate(order)}
        K = len(order); kmax = 2 * K
        for L in LAYERS:
            X, pids = read_last_token_per_seq(SHARD, L)
            keep = [i for i, p in enumerate(pids) if labels[p].get("family") == fam]
            Xf = X[keep].astype(np.float64)
            ci = np.array([idx[labels[pids[i]]["concept"]] for i in keep])
            cents = np.stack([Xf[ci == k].mean(0) for k in range(K)])
            for name, Xp in representations(Xf, cents).items():
                egK = eigengap_K(Xp, kmax)
                ari = float(adjusted_rand_score(
                    ci, KMeans(K, n_init=5, random_state=0).fit_predict(Xp)))
                results.append({"family": fam, "layer": L, "true_K": K,
                                "rep": name, "eigengap_K": egK, "ARI_at_trueK": ari})
            row = {r["rep"]: r["eigengap_K"] for r in results
                   if r["family"] == fam and r["layer"] == L}
            print(f"{fam:8} L{L:02d} K={K}: " +
                  " ".join(f"{k}={v}" for k, v in row.items()), flush=True)

    (OUT / "bin_count_repr_results.json").write_text(json.dumps(results, indent=2))

    reps = ["raw_pca50", "unitnorm", "zscore", "rm_top1", "rm_top2", "rm_top3",
            "save_embed"]
    print("\n=== layer-agnostic score: exact-K hits across all family x layer "
          "(max 12), and L40-only (max 2) ===")
    for rep in reps:
        rows = [r for r in results if r["rep"] == rep]
        exact = sum(r["eigengap_K"] == r["true_K"] for r in rows)
        near = sum(abs(r["eigengap_K"] - r["true_K"]) <= 1 for r in rows)
        l40 = sum(r["eigengap_K"] == r["true_K"] for r in rows if r["layer"] == 40)
        ari40 = np.mean([r["ARI_at_trueK"] for r in rows if r["layer"] == 40])
        print(f"  {rep:11}: exact {exact:2d}/12  within±1 {near:2d}/12  "
              f"| L40 exact {l40}/2  L40 meanARI {ari40:.2f}")
    print(f"\nwrote {OUT}/bin_count_repr_results.json")


if __name__ == "__main__":
    main()

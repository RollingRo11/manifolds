"""Sweep (layer × probe_day × n_bins × n_partner) for the lowest Procrustes
disparity vs supervised PCA.
"""
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
from scipy.spatial import procrustes
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent))
from load_acts import load_labels, read_last_token_per_seq
from sir_recovery import binary_probe_diff_of_means, slice_and_local_pca
from supervised import CALENDAR_ORDER

SHARD = "/data/artifacts/rohan/manifolds/concepts_olmo31_32b_v3/shard_dp00"
LABELS = "/home/rkathuria/manifolds/data/labels_v3.jsonl"
LAYERS = [16, 24, 32, 40, 48, 56]
PROBE_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
N_BINS_GRID = [10, 14, 20, 30]
N_PARTNER_GRID = [1, 2, 3]


def per_concept_centroids(emb, cidx, C):
    return np.stack([emb[cidx == k].mean(axis=0) for k in range(C)])


def nearest_centroid_accuracy(emb, cidx, C):
    cents = per_concept_centroids(emb, cidx, C)
    d2 = ((emb[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2)
    return float((np.argmin(d2, axis=1) == cidx).mean())


def main():
    labels = load_labels(Path(LABELS))
    order = CALENDAR_ORDER["weekday"]
    C = len(order)

    # Pre-load all layer activations for speed
    by_layer = {}
    for L in LAYERS:
        X_full, pids = read_last_token_per_seq(Path(SHARD), L)
        keep = [i for i, p in enumerate(pids) if labels[p]["family"] == "weekday"]
        X = X_full[keep]
        pids_w = [pids[i] for i in keep]
        cidx = np.array([order.index(labels[p]["concept"]) for p in pids_w])
        by_layer[L] = (X, pids_w, cidx)

    rows = []
    for L, day, nb, npart in product(LAYERS, PROBE_DAYS, N_BINS_GRID, N_PARTNER_GRID):
        X, pids_w, cidx = by_layer[L]
        sup_emb = PCA(n_components=1 + npart).fit_transform(X)
        sup_centroids = per_concept_centroids(sup_emb, cidx, C)
        pos_mask = np.array([labels[p]["concept"] == day for p in pids_w])
        w = binary_probe_diff_of_means(X, pos_mask)
        rec = slice_and_local_pca(X, w, n_bins=nb, n_partner=npart)
        rec_centroids = per_concept_centroids(rec.embedding, cidx, C)
        _, _, disp = procrustes(sup_centroids, rec_centroids)
        nn = nearest_centroid_accuracy(rec.embedding, cidx, C)
        rows.append({"layer": L, "probe_day": day, "n_bins": nb, "n_partner": npart,
                     "procrustes": float(disp), "nn_acc": nn,
                     "sup_nn": nearest_centroid_accuracy(sup_emb, cidx, C)})

    rows.sort(key=lambda r: r["procrustes"])
    Path("/home/rkathuria/manifolds/figures/sir_hparam_sweep.json").write_text(
        json.dumps(rows, indent=2))

    print("=" * 90)
    print(f"{'layer':>6} {'day':>10} {'n_bins':>7} {'n_part':>7} "
          f"{'NN-acc':>8} {'sup-NN':>8} {'procrustes':>12}")
    print("-" * 90)
    print("Top 20 lowest procrustes:")
    for r in rows[:20]:
        print(f"{r['layer']:>6} {r['probe_day']:>10} {r['n_bins']:>7} {r['n_partner']:>7} "
              f"{r['nn_acc']:>8.3f} {r['sup_nn']:>8.3f} {r['procrustes']:>12.4f}")
    print()
    print("Best per layer:")
    seen_L = set()
    for r in rows:
        if r["layer"] in seen_L: continue
        seen_L.add(r["layer"])
        print(f"  L{r['layer']}: probe={r['probe_day']:<10} n_bins={r['n_bins']:<3} "
              f"n_partner={r['n_partner']} → procrustes={r['procrustes']:.4f} "
              f"NN={r['nn_acc']:.3f} (sup={r['sup_nn']:.3f})")


if __name__ == "__main__":
    main()

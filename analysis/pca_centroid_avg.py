"""Cleaner PCA viz: average activations per (stem, concept) before PCA.

This denoises by averaging the ~3-5 prefix variations of each template,
reducing within-concept noise so the calendar-cyclic structure pops.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent))
from load_acts import load_labels, read_last_token_per_seq
from supervised import CALENDAR_ORDER


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir",
                    default="/data/artifacts/rohan/manifolds/concepts_olmo31_32b_v2/shard_dp00")
    ap.add_argument("--labels", default="/home/rkathuria/manifolds/data/labels_v2.jsonl")
    ap.add_argument("--out", default="/home/rkathuria/manifolds/figures/pca_weekday_centroid_avg.png")
    ap.add_argument("--family", default="weekday")
    ap.add_argument("--layers", default="16,24,32,40,48,56")
    ap.add_argument("--min-per-cell", type=int, default=2)
    args = ap.parse_args()

    labels = load_labels(Path(args.labels))
    layers = [int(x) for x in args.layers.split(",")]
    order = CALENDAR_ORDER[args.family]
    C = len(order)
    cmap = plt.get_cmap("hsv")

    fig = plt.figure(figsize=(5 * len(layers), 5))
    for li, L in enumerate(layers):
        X_full, pids = read_last_token_per_seq(Path(args.shard_dir), L)
        keep = [i for i, p in enumerate(pids) if labels[p]["family"] == args.family]
        X = X_full[keep]
        pids_f = [pids[i] for i in keep]

        # Group by (stem_idx, concept), average within group
        groups: dict[tuple[int, str], list[int]] = defaultdict(list)
        for i, p in enumerate(pids_f):
            groups[(labels[p]["stem_idx"], labels[p]["concept"])].append(i)
        rows = []
        rows_concept = []
        for (stem, concept), idxs in groups.items():
            if len(idxs) < args.min_per_cell:
                continue
            rows.append(X[idxs].mean(axis=0))
            rows_concept.append(concept)
        X_cent = np.stack(rows)
        cidx = np.array([order.index(c) for c in rows_concept])
        Xp = PCA(n_components=3).fit_transform(X_cent)

        ax = fig.add_subplot(1, len(layers), li + 1, projection="3d")
        for k in range(C):
            m = cidx == k
            ax.scatter(Xp[m, 0], Xp[m, 1], Xp[m, 2], s=30, alpha=0.85,
                       color=cmap(k / C), label=order[k])
        ax.set_title(f"layer {L}  (n={X_cent.shape[0]} cells)")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
        if li == 0:
            ax.legend(loc="upper left", fontsize=7, ncols=2)

    fig.suptitle(f"PCA of (stem, concept) means — denoised by averaging prefix paraphrases")
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

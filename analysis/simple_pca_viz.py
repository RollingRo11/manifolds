"""Bare-bones PCA visualization, paper-style.

  1. Load last-token activations from the v2 harvest.
  2. PCA to 3D.
  3. 3D scatter colored by concept.
"""
from __future__ import annotations

import argparse
import sys
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
    ap.add_argument("--out", default="/home/rkathuria/manifolds/figures/pca_layers.png")
    ap.add_argument("--family", default="weekday")
    ap.add_argument("--layers", default="16,24,32,40,48,56")
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
        cidx = np.array([order.index(labels[p]["concept"]) for p in pids_f])

        Xp = PCA(n_components=3).fit_transform(X)

        ax = fig.add_subplot(1, len(layers), li + 1, projection="3d")
        for k in range(C):
            m = cidx == k
            ax.scatter(Xp[m, 0], Xp[m, 1], Xp[m, 2], s=10, alpha=0.6,
                       color=cmap(k / C), label=order[k])
        ax.set_title(f"layer {L}")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
        if li == 0:
            ax.legend(loc="upper left", fontsize=7, ncols=2)

    fig.suptitle(f"PCA of last-token activations colored by held-out {args.family} label")
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

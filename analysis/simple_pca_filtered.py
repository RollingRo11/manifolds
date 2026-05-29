"""PCA viz on a *single template* of v2 data — minimizes prompt-content noise.

Filters by stem_idx so all prompts share the same scaffold, only the concept
varies (and optionally a few prefixes). Mirrors the paper's "one template,
many concepts" setup.
"""
from __future__ import annotations

import argparse
import json
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
    ap.add_argument("--out-dir", default="/home/rkathuria/manifolds/figures")
    ap.add_argument("--family", default="weekday")
    ap.add_argument("--layers", default="16,24,32,40,48,56")
    ap.add_argument("--stems", default="0",
                    help="Comma-separated stem_idx values to include (default: 0 only)")
    ap.add_argument("--prefixes", default="all",
                    help="'all' or comma-separated prefix_idx values")
    args = ap.parse_args()

    labels = load_labels(Path(args.labels))
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",")]
    keep_stems = set(int(s) for s in args.stems.split(","))
    keep_prefixes = None if args.prefixes == "all" else set(int(s) for s in args.prefixes.split(","))
    order = CALENDAR_ORDER[args.family]
    C = len(order)
    cmap = plt.get_cmap("hsv")

    suffix = f"stems{'_'.join(args.stems.split(','))}_pref{args.prefixes}"
    fig = plt.figure(figsize=(5 * len(layers), 5))
    for li, L in enumerate(layers):
        X_full, pids = read_last_token_per_seq(Path(args.shard_dir), L)
        keep_idx = []
        for i, p in enumerate(pids):
            lab = labels[p]
            if lab["family"] != args.family: continue
            if lab["stem_idx"] not in keep_stems: continue
            if keep_prefixes is not None and lab["prefix_idx"] not in keep_prefixes: continue
            keep_idx.append(i)
        X = X_full[keep_idx]
        pids_f = [pids[i] for i in keep_idx]
        cidx = np.array([order.index(labels[p]["concept"]) for p in pids_f])
        Xp = PCA(n_components=3).fit_transform(X)

        ax = fig.add_subplot(1, len(layers), li + 1, projection="3d")
        for k in range(C):
            m = cidx == k
            ax.scatter(Xp[m, 0], Xp[m, 1], Xp[m, 2], s=20, alpha=0.7,
                       color=cmap(k / C), label=order[k])
        ax.set_title(f"layer {L}  (n={X.shape[0]})")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
        if li == 0:
            ax.legend(loc="upper left", fontsize=7, ncols=2)

    fig.suptitle(f"PCA of last-token activations  (stems={args.stems}, prefixes={args.prefixes})")
    fig.tight_layout()
    out_path = out_dir / f"pca_{args.family}_{suffix}.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

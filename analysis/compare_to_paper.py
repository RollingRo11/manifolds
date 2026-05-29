"""Side-by-side comparison: supervised PCA vs probe-recovery at the best layer.

Paper benchmark (Wurgaft et al.): 7 weekday clusters arranged on a circle in
PCA space, layer-19 Llama-3.1-8B. We use OLMo-3.1-32B layer-16.

Output:
  Left panel  — supervised PCA top-3, paper's exact M_h reproduction
  Middle      — SIR-recovered (best positive probe = is_Wednesday)
  Right       — Procrustes alignment of the two centroid clouds
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import procrustes
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent))
from load_acts import load_labels, read_last_token_per_seq
from sir_recovery import binary_probe_diff_of_means, slice_and_local_pca
from supervised import CALENDAR_ORDER


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir",
                    default="/data/artifacts/rohan/manifolds/concepts_olmo31_32b_v3/shard_dp00")
    ap.add_argument("--labels", default="/home/rkathuria/manifolds/data/labels_v3.jsonl")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--probe-day", default="Wednesday")
    ap.add_argument("--out", default="/home/rkathuria/manifolds/figures/compare_to_paper.png")
    args = ap.parse_args()

    labels = load_labels(Path(args.labels))
    X_full, pids = read_last_token_per_seq(Path(args.shard_dir), args.layer)
    keep = [i for i, p in enumerate(pids) if labels[p]["family"] == "weekday"]
    X = X_full[keep]
    pids = [pids[i] for i in keep]
    order = CALENDAR_ORDER["weekday"]
    C = len(order)
    cidx = np.array([order.index(labels[p]["concept"]) for p in pids])
    cmap = plt.get_cmap("hsv")

    # --- Supervised PCA top-3 (paper's M_h analog) ---
    sup_emb = PCA(n_components=3).fit_transform(X)
    sup_centroids = np.stack([sup_emb[cidx == k].mean(axis=0) for k in range(C)])

    # --- SIR-recovery from is_<DAY> probe ---
    pos_mask = np.array([labels[p]["concept"] == args.probe_day for p in pids])
    w = binary_probe_diff_of_means(X, pos_mask)
    rec = slice_and_local_pca(X, w, n_bins=20, n_partner=2)
    rec_emb = rec.embedding
    rec_centroids = np.stack([rec_emb[cidx == k].mean(axis=0) for k in range(C)])

    # --- Procrustes alignment ---
    sup_std, rec_std, disparity = procrustes(sup_centroids, rec_centroids)
    print(f"Procrustes disparity (sup vs SIR-recovered) = {disparity:.4f}")
    # disparity 0 = identical, 1 = orthogonal

    fig = plt.figure(figsize=(18, 6))

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    for k in range(C):
        m = cidx == k
        ax1.scatter(sup_emb[m, 0], sup_emb[m, 1], sup_emb[m, 2],
                    s=18, alpha=0.55, color=cmap(k / C), label=order[k])
    loop = np.vstack([sup_centroids, sup_centroids[:1]])
    ax1.plot(loop[:, 0], loop[:, 1], loop[:, 2], "k-", lw=2.0)
    ax1.scatter(sup_centroids[:, 0], sup_centroids[:, 1], sup_centroids[:, 2],
                s=240, marker="o", facecolor="white", edgecolor="black",
                linewidth=2.5, zorder=10)
    for k in range(C):
        ax1.text(sup_centroids[k, 0], sup_centroids[k, 1], sup_centroids[k, 2],
                 order[k], fontsize=10, fontweight="bold")
    ax1.set_title(f"SUPERVISED PCA (paper-style M_h)\n"
                  f"OLMo-3.1-32B-Think  layer {args.layer}  n={X.shape[0]}",
                  fontsize=12)
    ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2"); ax1.set_zlabel("PC3")

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    for k in range(C):
        m = cidx == k
        ax2.scatter(rec_emb[m, 0], rec_emb[m, 1], rec_emb[m, 2],
                    s=18, alpha=0.55, color=cmap(k / C), label=order[k])
    loop = np.vstack([rec_centroids, rec_centroids[:1]])
    ax2.plot(loop[:, 0], loop[:, 1], loop[:, 2], "k-", lw=2.0)
    ax2.scatter(rec_centroids[:, 0], rec_centroids[:, 1], rec_centroids[:, 2],
                s=240, marker="o", facecolor="white", edgecolor="black",
                linewidth=2.5, zorder=10)
    for k in range(C):
        ax2.text(rec_centroids[k, 0], rec_centroids[k, 1], rec_centroids[k, 2],
                 order[k], fontsize=10, fontweight="bold")
    ax2.set_title(f"SIR-RECOVERED  (is_{args.probe_day} probe only)\n"
                  f"slice + per-bin PCA, n_bins=20",
                  fontsize=12)
    ax2.set_xlabel(f"s = ŵ·h"); ax2.set_ylabel("local tangent 1"); ax2.set_zlabel("local tangent 2")

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    for k in range(C):
        ax3.scatter(sup_std[k, 0], sup_std[k, 1], sup_std[k, 2],
                    s=240, marker="o", color=cmap(k / C), edgecolor="black",
                    linewidth=2.5, zorder=10, label=f"{order[k]} (sup)")
        ax3.scatter(rec_std[k, 0], rec_std[k, 1], rec_std[k, 2],
                    s=240, marker="X", color=cmap(k / C), edgecolor="black",
                    linewidth=2.5, zorder=10)
        # connecting line shows "how far" each supervised centroid moves to SIR-recovered
        ax3.plot([sup_std[k, 0], rec_std[k, 0]],
                 [sup_std[k, 1], rec_std[k, 1]],
                 [sup_std[k, 2], rec_std[k, 2]], "k-", alpha=0.4, lw=0.8)
    sup_loop = np.vstack([sup_std, sup_std[:1]])
    rec_loop = np.vstack([rec_std, rec_std[:1]])
    ax3.plot(sup_loop[:, 0], sup_loop[:, 1], sup_loop[:, 2], "-",
             color="black", lw=2.0, alpha=0.6, label="supervised loop")
    ax3.plot(rec_loop[:, 0], rec_loop[:, 1], rec_loop[:, 2], "--",
             color="black", lw=2.0, alpha=0.6, label="SIR-recovered loop")
    ax3.set_title(f"PROCRUSTES alignment\n"
                  f"disparity = {disparity:.4f}  (0=identical, 1=orthogonal)",
                  fontsize=12)
    ax3.set_xlabel("aligned dim 1"); ax3.set_ylabel("aligned dim 2"); ax3.set_zlabel("aligned dim 3")
    ax3.legend(loc="upper left", fontsize=7)

    fig.suptitle(f"OLMo-3.1-32B-Think weekday manifold:  "
                 f"supervised PCA (M_h)  vs.  single-probe SIR recovery   "
                 f"layer {args.layer}",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    plt.close(fig)
    print(f"wrote {args.out}")

    # Also report the rotation matrix Procrustes used
    # (helps show how much rigid alignment was needed)
    return disparity


if __name__ == "__main__":
    main()

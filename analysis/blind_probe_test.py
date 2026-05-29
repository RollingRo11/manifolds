"""Blind probe test.

Two questions:
  POS) Cross-set positive control. Train the chord probe on v2 (messy, multi-
       template) data; apply recovery on v3 (clean paper-template) activations.
       If the cycle is still recovered, the probe direction generalizes
       beyond its training distribution and the recovery algorithm doesn't
       need probe and data to come from the same dataset.

  NEG) Negative control. Apply an *unrelated* probe (refusal_v2, fiction-vs-
       real, testing_vs_conversation) to the same v3 weekday activations.
       The recovery should NOT find a calendar-ordered cycle — those probes
       have nothing to do with weekdays. If it still does, our recovery is
       hallucinating structure.

Both tests use the same v3 weekday data on the recovery side, so the
supervised-PCA "ground truth" is identical and the only thing changing is
the probe direction `w`.
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

sys.path.insert(0, str(Path(__file__).parent))
from load_acts import load_labels, read_last_token_per_seq
from manifold import reduce_pca
from recover_manifold import recover
from supervised import CALENDAR_ORDER
from visualize_compare import (
    chord_probe, cyclic_order_check, evaluate_centroids_on_circle,
    fit_circle_2d, restrict_to_family,
)


def find_best_pair(centroids_full, C):
    """Search over all axis pairs in the standardized centroid table."""
    cstd = (centroids_full - centroids_full.mean(axis=0)) / (centroids_full.std(axis=0) + 1e-9)
    n_axes = cstd.shape[1]
    best = None
    for i in range(n_axes):
        for j in range(i + 1, n_axes):
            cand = cstd[:, [i, j]]
            cx, cy, r, rms_rel = fit_circle_2d(cand)
            angles = np.arctan2(cand[:, 1] - cy, cand[:, 0] - cx)
            ok = cyclic_order_check(angles)
            score = (1 if ok else 0, -rms_rel)
            if best is None or score > best[0]:
                best = (score, (i, j), rms_rel, ok)
    return cstd, best


def run_recovery(X, w, cidx, C, label):
    """Run recovery, return (best_pair, rms, cycle_ok, recovered_centroids_2d)."""
    rec = recover(X, w, k_pca=64, k_nn=20, n_partner_axes=8)
    Y_full = rec.embedding
    centroids_full = np.stack([Y_full[cidx == k].mean(axis=0) for k in range(C)])
    cstd, best = find_best_pair(centroids_full, C)
    bi, bj = best[1]
    name_for = lambda k: "s" if k == 0 else f"v{k}"
    print(f"  [{label}] best pair = ({name_for(bi)}, {name_for(bj)})  "
          f"rms={best[2]:.3f}  cycle_ok={best[3]}")
    return cstd, best


def plot_summary(panels: list, out_path: Path, suptitle: str):
    n = len(panels)
    fig = plt.figure(figsize=(6 * n, 6))
    cmap = plt.get_cmap("hsv")
    for pi, (title, cstd, best, cidx, order) in enumerate(panels):
        ax = fig.add_subplot(1, n, pi + 1)
        bi, bj = best[1]
        cents = cstd[:, [bi, bj]]
        # color centroids
        for k, name in enumerate(order):
            ax.scatter(cents[k, 0], cents[k, 1], s=240, color=cmap(k / len(order)),
                       edgecolor="black", linewidth=1.5, zorder=10, label=name)
            ax.annotate(name, cents[k] + np.array([0.05, 0.05]), fontsize=9)
        # connect in calendar order
        loop = np.vstack([cents, cents[:1]])
        ax.plot(loop[:, 0], loop[:, 1], "k--", lw=1.2, alpha=0.6)
        # fit and draw circle
        cx, cy, r, rms_rel = fit_circle_2d(cents)
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(cx + r * np.cos(theta), cy + r * np.sin(theta), "b-", alpha=0.3)
        name_for = lambda k: "s" if k == 0 else f"v{k}"
        ax.set_title(f"{title}\nbest pair = ({name_for(bi)},{name_for(bj)})  "
                     f"cycle_ok={best[3]}  rms={best[2]:.3f}")
        ax.set_xlabel(name_for(bi)); ax.set_ylabel(name_for(bj))
        ax.set_aspect("equal", "datalim")
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2-shard",
                    default="/data/artifacts/rohan/manifolds/concepts_olmo31_32b_v2/shard_dp00")
    ap.add_argument("--v2-labels", default="/home/rkathuria/manifolds/data/labels_v2.jsonl")
    ap.add_argument("--v3-shard",
                    default="/data/artifacts/rohan/manifolds/concepts_olmo31_32b_v3/shard_dp00")
    ap.add_argument("--v3-labels", default="/home/rkathuria/manifolds/data/labels_v3.jsonl")
    ap.add_argument("--probes-dir", default="/data/artifacts/rohan/santi/probes")
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--out", default="/home/rkathuria/manifolds/figures/blind_probe_test.png")
    args = ap.parse_args()

    labels_v2 = load_labels(Path(args.v2_labels))
    labels_v3 = load_labels(Path(args.v3_labels))
    order = CALENDAR_ORDER["weekday"]
    C = len(order)

    # Load v2 and v3 weekday activations
    X_v2_all, pids_v2 = read_last_token_per_seq(Path(args.v2_shard), args.layer)
    X_v2, pids_v2 = restrict_to_family(X_v2_all, pids_v2, labels_v2, "weekday")
    X_v3_all, pids_v3 = read_last_token_per_seq(Path(args.v3_shard), args.layer)
    X_v3, pids_v3 = restrict_to_family(X_v3_all, pids_v3, labels_v3, "weekday")
    cidx_v3 = np.array([order.index(labels_v3[p]["concept"]) for p in pids_v3])
    print(f"v2 weekday n={X_v2.shape[0]}  v3 weekday n={X_v3.shape[0]}")

    panels = []

    # --- 0. SAME-SET BASELINE: probe trained on v3, recovered on v3 ---
    print("\n[same-set baseline]")
    w_v3 = chord_probe(X_v3, pids_v3, labels_v3, "Monday", "Friday")
    cstd, best = run_recovery(X_v3, w_v3, cidx_v3, C, "same-set v3 probe → v3")
    panels.append(("same-set: probe v3, data v3", cstd, best, cidx_v3, order))

    # --- 1. CROSS-SET POSITIVE: probe trained on v2, recovered on v3 ---
    print("\n[cross-set positive]")
    w_v2 = chord_probe(X_v2, pids_v2, labels_v2, "Monday", "Friday")
    cstd, best = run_recovery(X_v3, w_v2, cidx_v3, C, "v2 probe → v3 data")
    panels.append(("cross-set: probe v2, data v3", cstd, best, cidx_v3, order))

    # --- 2. NEGATIVE CONTROL: unrelated probe → v3 ---
    # Use refusal_direction_dense which has the layer we want
    refusal_path = Path(args.probes_dir) / "refusal_direction_dense.npz"
    if refusal_path.exists():
        npz = np.load(refusal_path, allow_pickle=False)
        key = f"L{args.layer}_refusal_direction"
        if key in npz.files:
            w_neg = npz[key].astype(np.float32)
            print("\n[negative control: refusal probe]")
            cstd, best = run_recovery(X_v3, w_neg, cidx_v3, C, "refusal probe → v3 data")
            panels.append((f"negative: refusal probe @ L{args.layer}, data v3",
                           cstd, best, cidx_v3, order))

    # --- 3. ANOTHER NEGATIVE: testing_vs_conversation probe → v3 ---
    tvc_path = Path(args.probes_dir) / "testing_vs_conversation_opus46_v2.npz"
    if tvc_path.exists():
        npz = np.load(tvc_path, allow_pickle=False)
        key = f"L{args.layer}_testing_vs_conversation_opus46"
        if key in npz.files:
            w_neg2 = npz[key].astype(np.float32)
            print("\n[negative control: testing-vs-conversation probe]")
            cstd, best = run_recovery(X_v3, w_neg2, cidx_v3, C, "testing-vs-conversation → v3 data")
            panels.append((f"negative: testing-vs-conversation probe @ L{args.layer}, data v3",
                           cstd, best, cidx_v3, order))

    plot_summary(panels, Path(args.out),
                 suptitle=f"Blind probe test  layer={args.layer}\n"
                          f"both ground-truth panels would be the supervised PCA on v3 (clean cycle)")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

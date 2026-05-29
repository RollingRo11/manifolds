"""Recover the cyclic weekday manifold using only a 'weekday-ness' probe.

  Step 1.  Train probe w = mean(weekday acts) - mean(month acts).  This is the
           single direction that captures "this activation is about weekdays
           rather than months". No within-weekday labels are used.

  Step 2.  Score every activation by s = w · h.  Keep the top-K most
           weekday-coded activations (without using day labels).

  Step 3.  PCA on those filtered activations.  The cyclic structure within
           the seven weekdays should appear, because the probe has isolated
           the weekday cluster from the broader activation space.

  Validation only: color the PCA scatter by held-out concept labels.

Compare against the paper-style supervised baseline: PCA on activations
that we *know* are from weekday prompts.
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


def make_concept_probe(X: np.ndarray, pids, labels, family_a: str, family_b: str):
    """Diff-of-means probe between two prompt categories."""
    A = np.array([i for i, p in enumerate(pids) if labels[p]["family"] == family_a])
    B = np.array([i for i, p in enumerate(pids) if labels[p]["family"] == family_b])
    w = X[A].mean(axis=0) - X[B].mean(axis=0)
    return w / (np.linalg.norm(w) + 1e-12), len(A), len(B)


def filter_by_stem(pids, labels, stems_keep: set[int]):
    keep = [i for i, p in enumerate(pids)
            if labels[p].get("stem_idx", -1) in stems_keep
            and labels[p]["family"] in ("weekday", "month")]
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir",
                    default="/data/artifacts/rohan/manifolds/concepts_olmo31_32b_v2/shard_dp00")
    ap.add_argument("--labels", default="/home/rkathuria/manifolds/data/labels_v2.jsonl")
    ap.add_argument("--out-dir", default="/home/rkathuria/manifolds/figures")
    ap.add_argument("--layers", default="16,24,32,40,48,56")
    # Filter prompts to a tight subset to reduce template noise
    ap.add_argument("--stems", default="0,11,12,36,37",
                    help="stem indices used as prompts (cleaner templates)")
    args = ap.parse_args()

    labels = load_labels(Path(args.labels))
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",")]
    stems_keep = set(int(s) for s in args.stems.split(","))

    weekday_order = CALENDAR_ORDER["weekday"]
    Cw = len(weekday_order)
    cmap = plt.get_cmap("hsv")

    fig = plt.figure(figsize=(5 * len(layers), 10))

    for li, L in enumerate(layers):
        X_full, pids_all = read_last_token_per_seq(Path(args.shard_dir), L)
        keep = filter_by_stem(pids_all, labels, stems_keep)
        X = X_full[keep]
        pids = [pids_all[i] for i in keep]
        family = np.array([labels[p]["family"] for p in pids])

        # Step 1: train weekday-vs-month probe
        w, nA, nB = make_concept_probe(X, pids, labels, "weekday", "month")
        s = X @ w
        # Step 2: pick top-K most weekday-coded activations (top half — no weekday count assumed)
        # We pretend we don't know how many weekdays there are; just take the
        # top half of probe scores.
        threshold = np.median(s)
        keep_mask = s > threshold
        X_recover = X[keep_mask]
        pids_recover = [pids[i] for i, m in enumerate(keep_mask) if m]
        # ground-truth weekday labels for coloring (post-hoc only)
        is_weekday_gt = np.array([labels[p]["family"] == "weekday" for p in pids_recover])
        cidx_recover = np.array([
            weekday_order.index(labels[p]["concept"]) if labels[p]["family"] == "weekday" else -1
            for p in pids_recover
        ])

        # Step 3: PCA on probe-filtered activations
        Xp_recover = PCA(n_components=3).fit_transform(X_recover)

        ax_top = fig.add_subplot(2, len(layers), li + 1, projection="3d")
        # Plot non-weekday filtered points in grey, weekday points colored
        m_nw = ~is_weekday_gt
        ax_top.scatter(Xp_recover[m_nw, 0], Xp_recover[m_nw, 1], Xp_recover[m_nw, 2],
                       s=12, alpha=0.25, color="lightgray", label="(non-weekday filtered in)")
        for k in range(Cw):
            mk = (cidx_recover == k)
            ax_top.scatter(Xp_recover[mk, 0], Xp_recover[mk, 1], Xp_recover[mk, 2],
                           s=18, alpha=0.75, color=cmap(k / Cw), label=weekday_order[k])
        recall = float(is_weekday_gt.mean())
        ax_top.set_title(f"layer {L}: PCA on probe-filtered\nweekday recall in top half = {recall:.2f}")
        ax_top.set_xlabel("PC1"); ax_top.set_ylabel("PC2"); ax_top.set_zlabel("PC3")
        if li == 0:
            ax_top.legend(loc="upper left", fontsize=6, ncols=2)

        # Bottom row: paper-style supervised baseline (PCA on known weekday acts only)
        keep_w = [i for i, p in enumerate(pids) if labels[p]["family"] == "weekday"]
        Xw = X[keep_w]
        pids_w = [pids[i] for i in keep_w]
        cidx_w = np.array([weekday_order.index(labels[p]["concept"]) for p in pids_w])
        Xpw = PCA(n_components=3).fit_transform(Xw)

        ax_bot = fig.add_subplot(2, len(layers), len(layers) + li + 1, projection="3d")
        for k in range(Cw):
            mk = cidx_w == k
            ax_bot.scatter(Xpw[mk, 0], Xpw[mk, 1], Xpw[mk, 2],
                           s=18, alpha=0.75, color=cmap(k / Cw), label=weekday_order[k])
        ax_bot.set_title(f"layer {L}: PCA on known-weekday acts (paper baseline)")
        ax_bot.set_xlabel("PC1"); ax_bot.set_ylabel("PC2"); ax_bot.set_zlabel("PC3")

    fig.suptitle(f"top: probe-driven recovery   bottom: supervised baseline\n"
                 f"probe = mean(weekday) − mean(month)   stems={args.stems}")
    fig.tight_layout()
    out_path = out_dir / f"recover_via_probe_stems{args.stems.replace(',', '_')}.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

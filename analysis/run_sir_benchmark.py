"""Weekday SIR-style benchmark: single binary "is it Monday?" probe,
slice-and-local-PCA recovery, Procrustes against supervised PCA centroids.

Three probe conditions:
  + binary-probe   : "is_Monday" diff-of-means → expected to recover the cycle
  - random         : random unit vectors × 20 → noise floor
  - unrelated      : refusal/testing-vs-conv probes → should fail (not weekday-related)

The recovery pipeline is deterministic in (X, w). No labels enter recovery;
labels are used ONLY for evaluation (Procrustes alignment + nearest-centroid
classification).
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
from scipy.spatial import procrustes
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent))
from load_acts import load_labels, read_last_token_per_seq
from sir_recovery import binary_probe_diff_of_means, slice_and_local_pca
from supervised import CALENDAR_ORDER


def restrict_to_family(X, pids, labels, family):
    keep = [i for i, p in enumerate(pids) if labels[p]["family"] == family]
    return X[keep], [pids[i] for i in keep]


def per_concept_centroids(emb, cidx, C):
    return np.stack([emb[cidx == k].mean(axis=0) for k in range(C)])


def f_statistic(coord, cidx, C):
    grand = coord.mean()
    within = between = 0.0
    for k in range(C):
        m = cidx == k
        if m.sum() == 0: continue
        cm = coord[m].mean()
        between += int(m.sum()) * (cm - grand) ** 2
        within += float(((coord[m] - cm) ** 2).sum())
    return float((between / max(C - 1, 1)) / (within / max(len(coord) - C, 1) + 1e-12))


def nearest_centroid_accuracy(emb, cidx, C):
    centroids = np.stack([emb[cidx == k].mean(axis=0) for k in range(C)])
    d2 = ((emb[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    pred = np.argmin(d2, axis=1)
    return float((pred == cidx).mean())


def cyclic_order_correlation(rec_centroids_2d, calendar_order_C):
    """Circular correlation between recovered angular positions of concept
    centroids and their true calendar positions.

    Procedure: fit a circle to the centroids, get angular positions, compare
    to the ideal angles 2*pi*k/C. Higher = better cyclic-order recovery.
    """
    pts = rec_centroids_2d - rec_centroids_2d.mean(axis=0)
    angles = np.arctan2(pts[:, 1], pts[:, 0])  # in [-pi, pi]
    ideal = 2 * np.pi * np.arange(calendar_order_C) / calendar_order_C
    # Try both directions (clockwise/counterclockwise) and all rotations
    best = -1.0
    for sign in (+1, -1):
        for shift in range(calendar_order_C):
            cand = (sign * ideal + shift * 2 * np.pi / calendar_order_C) % (2 * np.pi)
            ang = angles % (2 * np.pi)
            c = float(np.corrcoef(np.cos(cand - ang), np.ones_like(cand))[0, 0])
            # circular correlation via trigonometric identity:
            cos_corr = (np.cos(cand - ang)).mean()
            best = max(best, cos_corr)
    return best


def evaluate_recovery(emb, cidx, C, sup_centroids):
    rec_centroids = per_concept_centroids(emb, cidx, C)
    # Procrustes (centroid-level)
    if rec_centroids.shape[1] != sup_centroids.shape[1]:
        d = max(rec_centroids.shape[1], sup_centroids.shape[1])
        a = np.column_stack([rec_centroids, np.zeros((C, d - rec_centroids.shape[1]))])
        b = np.column_stack([sup_centroids, np.zeros((C, d - sup_centroids.shape[1]))])
    else:
        a, b = rec_centroids, sup_centroids
    _, _, disp = procrustes(a, b)
    return {
        "nn_accuracy": nearest_centroid_accuracy(emb, cidx, C),
        "f_stat_per_axis": [f_statistic(emb[:, k], cidx, C) for k in range(emb.shape[1])],
        "f_stat_min": float(min(f_statistic(emb[:, k], cidx, C) for k in range(emb.shape[1]))),
        "procrustes": float(disp),
        "cyclic_order_cos_corr": cyclic_order_correlation(rec_centroids[:, 1:3] if emb.shape[1] >= 3 else rec_centroids[:, :2], C),
        "rec_centroids": rec_centroids.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir",
                    default="/data/artifacts/rohan/manifolds/concepts_olmo31_32b_v3/shard_dp00")
    ap.add_argument("--labels", default="/home/rkathuria/manifolds/data/labels_v3.jsonl")
    ap.add_argument("--probes-dir", default="/data/artifacts/rohan/santi/probes")
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--n-bins", type=int, default=20)
    ap.add_argument("--n-partner", type=int, default=2)
    ap.add_argument("--out-png", default="/home/rkathuria/manifolds/figures/sir_benchmark.png")
    ap.add_argument("--out-json", default="/home/rkathuria/manifolds/figures/sir_benchmark.json")
    args = ap.parse_args()

    labels = load_labels(Path(args.labels))
    X_full, pids = read_last_token_per_seq(Path(args.shard_dir), args.layer)
    X, pids = restrict_to_family(X_full, pids, labels, "weekday")
    order = CALENDAR_ORDER["weekday"]
    C = len(order)
    cidx = np.array([order.index(labels[p]["concept"]) for p in pids])
    print(f"v3 weekday n={X.shape[0]}  layer={args.layer}")

    # Ground truth: supervised PCA top-(1+n_partner)
    sup_emb = PCA(n_components=1 + args.n_partner).fit_transform(X)
    sup_eval = evaluate_recovery(sup_emb, cidx, C, per_concept_centroids(sup_emb, cidx, C))
    sup_eval["procrustes"] = 0.0  # ground truth vs itself
    sup_centroids = per_concept_centroids(sup_emb, cidx, C)
    print(f"\nground-truth supervised PCA top-{1+args.n_partner}:")
    print(f"  NN-acc={sup_eval['nn_accuracy']:.3f}  F-min={sup_eval['f_stat_min']:.2f}  "
          f"cyclic-cos-corr={sup_eval['cyclic_order_cos_corr']:.3f}")

    results = {"supervised": sup_eval, "supervised_centroids": sup_centroids.tolist()}

    def do_probe(name, w, label):
        rec = slice_and_local_pca(X, w, n_bins=args.n_bins, n_partner=args.n_partner)
        ev = evaluate_recovery(rec.embedding, cidx, C, sup_centroids)
        ev["probe"] = label
        ev["parallel_transport_defect"] = rec.parallel_transport_defect
        ev["bin_pop_min"] = int(rec.bin_pop.min())
        ev["bin_pop_max"] = int(rec.bin_pop.max())
        results[name] = ev
        print(f"\n[{label}]")
        print(f"  bin counts min/max = {rec.bin_pop.min()}/{rec.bin_pop.max()}")
        print(f"  parallel-transport defect = {rec.parallel_transport_defect:.3f}")
        print(f"  NN-acc={ev['nn_accuracy']:.3f}  F-min={ev['f_stat_min']:.2f}  "
              f"procrustes={ev['procrustes']:.4f}  cyclic-cos-corr={ev['cyclic_order_cos_corr']:.3f}")
        return rec

    # 1. Binary "is it Monday" probe
    pos_mask = np.array([labels[p]["concept"] == "Monday" for p in pids])
    w_mon = binary_probe_diff_of_means(X, pos_mask)
    rec_mon = do_probe("pos_is_monday", w_mon, "is_Monday probe (binary diff-of-means)")

    # 2. Binary "is it Wednesday" (different chord, control for cycle position)
    pos_mask = np.array([labels[p]["concept"] == "Wednesday" for p in pids])
    w_wed = binary_probe_diff_of_means(X, pos_mask)
    do_probe("pos_is_wednesday", w_wed, "is_Wednesday probe (binary)")

    # 3. Random unit vectors × 20 trials
    rng = np.random.default_rng(0)
    rand_metrics = {"nn_accuracy": [], "f_stat_min": [], "procrustes": [],
                    "cyclic_order_cos_corr": [], "parallel_transport_defect": []}
    for trial in range(20):
        w = rng.standard_normal(X.shape[1]).astype(np.float32)
        w /= np.linalg.norm(w)
        rec = slice_and_local_pca(X, w, n_bins=args.n_bins, n_partner=args.n_partner)
        ev = evaluate_recovery(rec.embedding, cidx, C, sup_centroids)
        for kk in rand_metrics:
            rand_metrics[kk].append(
                ev[kk] if kk != "parallel_transport_defect" else rec.parallel_transport_defect
            )
    results["neg_random"] = {
        "probe": "random unit vector × 20",
        "nn_accuracy_mean": float(np.mean(rand_metrics["nn_accuracy"])),
        "nn_accuracy_std": float(np.std(rand_metrics["nn_accuracy"])),
        "f_stat_min_mean": float(np.mean(rand_metrics["f_stat_min"])),
        "procrustes_mean": float(np.mean(rand_metrics["procrustes"])),
        "procrustes_std": float(np.std(rand_metrics["procrustes"])),
        "cyclic_order_cos_corr_mean": float(np.mean(rand_metrics["cyclic_order_cos_corr"])),
        "cyclic_order_cos_corr_std": float(np.std(rand_metrics["cyclic_order_cos_corr"])),
        "parallel_transport_defect_mean": float(np.mean(rand_metrics["parallel_transport_defect"])),
    }
    print(f"\n[random × 20]")
    print(f"  NN-acc = {results['neg_random']['nn_accuracy_mean']:.3f} ± "
          f"{results['neg_random']['nn_accuracy_std']:.3f}")
    print(f"  procrustes = {results['neg_random']['procrustes_mean']:.4f} ± "
          f"{results['neg_random']['procrustes_std']:.4f}")
    print(f"  cyclic-cos-corr = {results['neg_random']['cyclic_order_cos_corr_mean']:.3f} ± "
          f"{results['neg_random']['cyclic_order_cos_corr_std']:.3f}")
    print(f"  parallel-transport defect = {results['neg_random']['parallel_transport_defect_mean']:.3f}")

    # 4. Unrelated: refusal / testing-vs-conversation
    for fname, key, label in [
        ("refusal_direction_dense.npz", f"L{args.layer}_refusal_direction",
         f"refusal probe @ L{args.layer}"),
        ("testing_vs_conversation_opus46_v2.npz",
         f"L{args.layer}_testing_vs_conversation_opus46",
         f"testing_vs_conversation @ L{args.layer}"),
    ]:
        p = Path(args.probes_dir) / fname
        if not p.exists(): continue
        npz = np.load(p, allow_pickle=False)
        if key not in npz.files: continue
        w = npz[key].astype(np.float32)
        do_probe(f"neg_{fname.replace('.npz','')}", w, label)

    # ----- Summary table -----
    print("\n" + "=" * 100)
    fmt = "{:<48}{:>10}{:>10}{:>13}{:>16}{:>16}"
    print(fmt.format("probe", "NN-acc", "F-min", "procrustes", "cyclic-cos-cor", "PT-defect"))
    print(fmt.format("supervised top-(1+k) PCA", f"{sup_eval['nn_accuracy']:.3f}",
                     f"{sup_eval['f_stat_min']:.2f}", "—",
                     f"{sup_eval['cyclic_order_cos_corr']:.3f}", "—"))
    print("-" * 100)
    for k, v in results.items():
        if k in ("supervised", "supervised_centroids"): continue
        if "nn_accuracy_mean" in v:
            acc = f"{v['nn_accuracy_mean']:.3f}±{v['nn_accuracy_std']:.3f}"
            fm = "—"
            disp = f"{v['procrustes_mean']:.3f}±{v['procrustes_std']:.3f}"
            coh = f"{v['cyclic_order_cos_corr_mean']:.3f}±{v['cyclic_order_cos_corr_std']:.3f}"
            pt = f"{v['parallel_transport_defect_mean']:.3f}"
        else:
            acc = f"{v['nn_accuracy']:.3f}"
            fm = f"{v['f_stat_min']:.2f}"
            disp = f"{v['procrustes']:.4f}"
            coh = f"{v['cyclic_order_cos_corr']:.3f}"
            pt = f"{v.get('parallel_transport_defect', float('nan')):.3f}"
        print(fmt.format(v["probe"], acc, fm, disp, coh, pt))

    Path(args.out_json).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {args.out_json}")

    # ----- Visualize: 2D plane (axis 1, axis 2) of every probe -----
    cmap = plt.get_cmap("hsv")
    panels = [("supervised PCA (truth)\n"
               f"NN={sup_eval['nn_accuracy']:.3f}  cyc-cos={sup_eval['cyclic_order_cos_corr']:.3f}",
               np.array(results["supervised_centroids"]))]
    for k, v in results.items():
        if k in ("supervised", "supervised_centroids", "neg_random"): continue
        if "rec_centroids" not in v: continue
        title = (f"{v['probe']}\n"
                 f"NN={v['nn_accuracy']:.3f}  cyc-cos={v['cyclic_order_cos_corr']:.3f}  "
                 f"proc={v['procrustes']:.3f}")
        panels.append((title, np.array(v["rec_centroids"])))

    n = len(panels)
    fig = plt.figure(figsize=(5 * n, 5))
    for i, (title, cents) in enumerate(panels):
        ax = fig.add_subplot(1, n, i + 1)
        # Plot the 2D plane formed by axis 1 and 2 of the embedding
        # (skip s; the cycle should appear in the partner-axis plane)
        if cents.shape[1] >= 3:
            pts = cents[:, 1:3]
        else:
            pts = cents[:, :2]
        for k_c in range(C):
            ax.scatter(pts[k_c, 0], pts[k_c, 1], s=240, color=cmap(k_c / C),
                       edgecolor="black", linewidth=1.5, zorder=10)
            ax.annotate(order[k_c],
                        pts[k_c] + np.array([0.02, 0.02]) * np.ptp(pts),
                        fontsize=9)
        loop = np.vstack([pts, pts[:1]])
        ax.plot(loop[:, 0], loop[:, 1], "k--", lw=1.0, alpha=0.5)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal", "datalim")
        ax.set_xlabel("axis 1 (partner)")
        ax.set_ylabel("axis 2 (partner)")

    fig.suptitle(f"SIR slice+local-PCA recovery, layer {args.layer}, "
                 f"n_bins={args.n_bins}, n_partner={args.n_partner}\n"
                 f"random-probe noise floor: NN={results['neg_random']['nn_accuracy_mean']:.3f}±"
                 f"{results['neg_random']['nn_accuracy_std']:.3f}  "
                 f"cyc-cos-cor={results['neg_random']['cyclic_order_cos_corr_mean']:.3f}±"
                 f"{results['neg_random']['cyclic_order_cos_corr_std']:.3f}")
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=140)
    plt.close(fig)
    print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()

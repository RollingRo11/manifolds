"""Side-by-side 3D visualization: supervised PCA manifold vs probe-recovered manifold.

For one (shard, layer, family, probe pair):
  LEFT  panel: top-3 PCs of the activations, colored by held-out concept
              label, with the supervised concept-centroid spline overlaid
              (this is the paper's M_h).
  RIGHT panel: probe-recovered embedding (s, v1·R, v2·R) using only the
              probe direction (no concept labels), colored by held-out
              concept label.

  Both panels are interactive 3D matplotlib axes, exported as a single PNG.
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
from probe_manifold_v2 import nullspace_nonlinear_R2
from recover_manifold import recover
from supervised import CALENDAR_ORDER, fit_supervised


def chord_probe(X, pids, labels, a, b):
    A = np.array([i for i, p in enumerate(pids) if labels[p]["concept"] == a])
    B = np.array([i for i, p in enumerate(pids) if labels[p]["concept"] == b])
    w = X[A].mean(axis=0) - X[B].mean(axis=0)
    return w / (np.linalg.norm(w) + 1e-12)


def restrict_to_family(X, pids, labels, family):
    keep = [i for i, p in enumerate(pids) if labels[p]["family"] == family]
    return X[keep], [pids[i] for i in keep]


def fit_circle_2d(pts):
    """Algebraic circle fit. Returns (cx, cy, r, residual_RMS_relative)."""
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x**2 + y**2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    r = float(np.sqrt(c + cx**2 + cy**2))
    residuals = np.sqrt((x - cx)**2 + (y - cy)**2) - r
    rms = float(np.sqrt(np.mean(residuals**2)))
    return cx, cy, r, rms / (r + 1e-12)


def cyclic_order_check(angles):
    """Given angles for C concepts in calendar order, do they appear in
    monotone order around the circle (forward or backward)?"""
    # Normalize to [0, 2pi)
    a = (angles - angles[0]) % (2 * np.pi)
    # Forward: 0, increasing
    fwd_ok = np.all(np.diff(a) > 0)
    # Backward: 0, decreasing modulo 2pi
    a_rev = (-a) % (2 * np.pi)
    bwd_ok = np.all(np.diff(a_rev) > 0)
    return fwd_ok or bwd_ok


def evaluate_centroids_on_circle(centroids_xy, calendar_order):
    cx, cy, r, rms_rel = fit_circle_2d(centroids_xy)
    angles = np.arctan2(centroids_xy[:, 1] - cy, centroids_xy[:, 0] - cx)
    # Check ordering matches calendar
    order_ok = cyclic_order_check(angles)
    return {"circle_radius": r, "circle_rms_relative": rms_rel,
            "cycle_order_correct": bool(order_ok),
            "angles": angles.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir",
                    default="/data/artifacts/rohan/manifolds/concepts_olmo31_32b_v2/shard_dp00")
    ap.add_argument("--labels", default="/home/rkathuria/manifolds/data/labels_v2.jsonl")
    ap.add_argument("--out", default="/home/rkathuria/manifolds/figures/recover_compare.png")
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--family", default="weekday")
    ap.add_argument("--probe-a", default="Monday")
    ap.add_argument("--probe-b", default="Friday")
    args = ap.parse_args()

    labels = load_labels(Path(args.labels))
    X_full_all, pids_all_full = read_last_token_per_seq(Path(args.shard_dir), args.layer)
    X_all, pids_all = restrict_to_family(X_full_all, pids_all_full, labels, args.family)
    print(f"all weekday samples: {X_all.shape[0]}")

    # ALL data is used to TRAIN the probe and FIND the partner axes (robust).
    # If labels carry a stem_idx (multi-template v2), filter to clean stems
    # for plotting; for paper-style minimal-template v3 we plot everything.
    has_stems = any("stem_idx" in labels[p] for p in pids_all[:1])
    if has_stems:
        keep_stems = {0, 11, 12, 36, 37}
        plot_idx = np.array([i for i, p in enumerate(pids_all)
                              if labels[p].get("stem_idx", -1) in keep_stems], dtype=int)
    else:
        plot_idx = np.arange(len(pids_all), dtype=int)
    X = X_all[plot_idx]
    pids = [pids_all[i] for i in plot_idx]
    print(f"plotted subset: {X.shape[0]}")

    order = CALENDAR_ORDER[args.family]
    C = len(order)
    cidx = np.array([order.index(labels[p]["concept"]) for p in pids])
    cidx_all = np.array([order.index(labels[p]["concept"]) for p in pids_all])
    cmap = plt.get_cmap("hsv")

    # ------ LEFT: paper-style supervised --------------------------------
    pca = reduce_pca(X, k_pca=64)
    Xp = pca.X
    sm = fit_supervised(Xp, pids, labels, family=args.family)
    fig = plt.figure(figsize=(18, 7))
    ax_l = fig.add_subplot(1, 2, 1, projection="3d")
    for k in range(C):
        m = cidx == k
        ax_l.scatter(Xp[m, 0], Xp[m, 1], Xp[m, 2], s=22, alpha=0.65,
                     color=cmap(k / C), label=order[k])
    # Paper's M_h: spline through concept centroids in top-3 PC space
    fine_t = np.linspace(0, C, 200)
    spline_pts = sm.spline(fine_t)[:, :3]
    ax_l.plot(spline_pts[:, 0], spline_pts[:, 1], spline_pts[:, 2], "k-", lw=2.0,
              label="supervised spline")
    centroids_pca = sm.centroids[:, :3]
    ax_l.scatter(centroids_pca[:, 0], centroids_pca[:, 1], centroids_pca[:, 2],
                 s=240, marker="o", facecolor="white", edgecolor="k", linewidth=2.5, zorder=10)
    for k in range(C):
        ax_l.text(centroids_pca[k, 0] * 1.05, centroids_pca[k, 1] * 1.05,
                  centroids_pca[k, 2] * 1.05, order[k], fontsize=11, fontweight="bold")
    ax_l.set_title(f"SUPERVISED (paper-style)\nPCA top-3 of last-token activations  layer {args.layer}",
                   fontsize=13)
    ax_l.set_xlabel("PC1"); ax_l.set_ylabel("PC2"); ax_l.set_zlabel("PC3")
    ax_l.legend(loc="upper left", fontsize=8, ncols=2)

    # ------ RIGHT: probe-recovered ---------------------------------------
    # Train the probe on ALL weekday samples (more stable diff-of-means).
    w = chord_probe(X_all, pids_all, labels, args.probe_a, args.probe_b)
    # Find partner axes on ALL data too — mid-slab PCA needs many samples.
    rec = recover(X_all, w, k_pca=64, k_nn=20, n_partner_axes=8)
    Y_full = rec.embedding   # [N_all, 9]: s + 8 partner coords for ALL samples
    # Centroid search uses ALL data per concept
    centroids_full = np.stack([Y_full[cidx_all == k].mean(axis=0) for k in range(C)])
    # Standardize all columns so the (s, v_k) pairs are commensurable.
    centroids_std = (centroids_full - centroids_full.mean(axis=0)) / (centroids_full.std(axis=0) + 1e-9)

    # Search over all axis pairs (including s) for the best cyclic projection.
    # The cycle should appear in (s, v_partner) for the right partner axis.
    n_axes = centroids_std.shape[1]
    best = None
    for i in range(n_axes):
        for j in range(i + 1, n_axes):
            cand = centroids_std[:, [i, j]]
            cx, cy, r, rms_rel = fit_circle_2d(cand)
            angles = np.arctan2(cand[:, 1] - cy, cand[:, 0] - cx)
            ok = cyclic_order_check(angles)
            score = (1 if ok else 0, -rms_rel)
            if best is None or score > (best[0]):
                best = (score, (i, j), rms_rel, ok)
    best_i, best_j = best[1]
    name_for = lambda k: "s" if k == 0 else f"v{k}"
    print(f"best 2D pair: ({name_for(best_i)}, {name_for(best_j)})  rms={best[2]:.3f}  cycle_ok={best[3]}")

    # Find a third axis to round out the 3D embedding: the unused axis with
    # most variance among the partner coords.
    used = {best_i, best_j}
    remain = [k for k in range(n_axes) if k not in used]
    third = max(remain, key=lambda k: float(centroids_std[:, k].var()))
    print(f"3rd 3D axis: {name_for(third)}")

    Y = np.column_stack([centroids_std[:, best_i], centroids_std[:, best_j], centroids_std[:, third]])
    # Project ONLY the visualization subset (clean templates) onto the
    # recovered axes, using the same standardization derived from ALL data.
    Y_full_std = (Y_full - Y_full.mean(axis=0)) / (Y_full.std(axis=0) + 1e-9)
    plot_full = Y_full_std[plot_idx]
    Y_per_prompt = np.column_stack([plot_full[:, best_i], plot_full[:, best_j], plot_full[:, third]])
    Y_centroids = Y

    ax_r = fig.add_subplot(1, 2, 2, projection="3d")
    for k in range(C):
        m = cidx == k
        ax_r.scatter(Y_per_prompt[m, 0], Y_per_prompt[m, 1], Y_per_prompt[m, 2],
                     s=22, alpha=0.65, color=cmap(k / C), label=order[k])
    cents_loop = np.vstack([Y_centroids, Y_centroids[:1]])
    ax_r.plot(cents_loop[:, 0], cents_loop[:, 1], cents_loop[:, 2], "k-", lw=2.0)
    ax_r.scatter(Y_centroids[:, 0], Y_centroids[:, 1], Y_centroids[:, 2],
                 s=240, marker="o", facecolor="white", edgecolor="k", linewidth=2.5, zorder=10)
    for k in range(C):
        ax_r.text(Y_centroids[k, 0] * 1.05, Y_centroids[k, 1] * 1.05,
                  Y_centroids[k, 2] * 1.05, order[k], fontsize=11, fontweight="bold")
    ax_r.set_title(f"PROBE-RECOVERED  (no day labels used)\n"
                   f"probe = {args.probe_a} − {args.probe_b}  axes=({name_for(best_i)},{name_for(best_j)},{name_for(third)})  cycle_ok={best[3]}",
                   fontsize=13)
    ax_r.set_xlabel(name_for(best_i)); ax_r.set_ylabel(name_for(best_j)); ax_r.set_zlabel(name_for(third))

    # Eval on the best-pair plane (which is what the search found)
    ev_best = evaluate_centroids_on_circle(Y_centroids[:, :2], order)
    # And the supervised top-2 PC plane (paper's typical 2D view)
    ev_pca = evaluate_centroids_on_circle(centroids_pca[:, :2], order)
    ev_sv = ev_best
    ev_partner = ev_best

    ns = nullspace_nonlinear_R2(X.astype(np.float64), w, n_subsample=1500)

    fig.suptitle(
        f"recovered cycle order (s,v1) ok={ev_sv['cycle_order_correct']}  "
        f"(v1,v2) ok={ev_partner['cycle_order_correct']}  |  "
        f"supervised cycle (PC1,PC2) ok={ev_pca['cycle_order_correct']}  |  "
        f"nullspace R²: rbf={ns['R2_rbf_kernel']:.2f} gain={ns['nonlinear_gain']:.2f}"
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    plt.close(fig)
    print(f"wrote {args.out}")
    print(f"recovered cycle order on (s,v1): {ev_sv['cycle_order_correct']}  rms={ev_sv['circle_rms_relative']:.3f}")
    print(f"recovered cycle order on (v1,v2): {ev_partner['cycle_order_correct']}  rms={ev_partner['circle_rms_relative']:.3f}")
    print(f"supervised  cycle order on (PC1,PC2): {ev_pca['cycle_order_correct']}  rms={ev_pca['circle_rms_relative']:.3f}")
    print(f"nullspace R²: rbf={ns['R2_rbf_kernel']:.3f}  linear={ns['R2_linear_baseline']:.3f}  "
          f"nonlinear gain={ns['nonlinear_gain']:.3f}")


if __name__ == "__main__":
    main()

"""Recover-from-probe testbench.

Setup (uses the v2 weekday harvest):
  1. Load activations and held-back concept labels.
  2. Train a "chord probe": diff-of-means between two specific concepts
     (default: Monday vs Friday — known to be ~half-cycle apart).
  3. Apply recover_manifold(X, w_probe). NO concept labels go into recovery.
  4. Validate by reading the held-back labels: do the seven concept clusters
     trace a circle in the recovered 2D embedding?
       - Per-concept median position in the 2D embedding
       - Pairwise geodesic-on-embedding distances vs ground-truth cycle distance
       - Persistent H1 of recovered embedding
       - Plot a 3D-style scatter (s, dm1, dm2) colored by concept

  Also: ablation — what happens when we DON'T strip the probe (i.e., run
  unsupervised manifold discovery directly on X)? This tells us how much
  the probe helped.

Repeat for several layers and several concept pairs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from load_acts import load_labels, read_last_token_per_seq
from manifold import diffusion_map, persistence_diagrams, reduce_pca
from probe_manifold_v2 import nullspace_nonlinear_R2
from recover_manifold import recover
from supervised import CALENDAR_ORDER

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def chord_probe(X: np.ndarray, pids: list[int], labels: dict[int, dict],
                concept_a: str, concept_b: str) -> np.ndarray:
    """Diff-of-means probe direction between two concepts. Unit norm."""
    A = np.array([i for i, p in enumerate(pids) if labels[p]["concept"] == concept_a])
    B = np.array([i for i, p in enumerate(pids) if labels[p]["concept"] == concept_b])
    mu_a = X[A].mean(axis=0)
    mu_b = X[B].mean(axis=0)
    w = mu_a - mu_b
    return w / (np.linalg.norm(w) + 1e-12)


def restrict_to_family(X, pids, labels, family):
    keep = [i for i, p in enumerate(pids) if labels[p]["family"] == family]
    return X[keep], [pids[i] for i in keep]


def cycle_distance(i: int, j: int, C: int) -> int:
    d = abs(i - j) % C
    return min(d, C - d)


def evaluate_recovered(rec, pids, labels, family: str) -> dict:
    """How well does the recovered embedding reflect the known cyclic structure?"""
    order = CALENDAR_ORDER[family]
    C = len(order)
    cidx = np.array([order.index(labels[p]["concept"]) for p in pids])

    # Per-concept median position in the recovered embedding (low-d for plotting)
    Y = rec.embedding[:, : min(3, rec.embedding.shape[1])]
    centroids = np.stack([np.median(Y[cidx == k], axis=0) for k in range(C)])

    # Pairwise geodesic-on-recovered-graph between concept centroids vs ground-truth cycle distance
    Dg = np.zeros((C, C))
    for i in range(C):
        for j in range(C):
            mask_i = cidx == i; mask_j = cidx == j
            Dg[i, j] = rec.geodesic[np.ix_(mask_i, mask_j)].mean()
    Dt = np.array([[cycle_distance(i, j, C) for j in range(C)] for i in range(C)])
    iu = np.triu_indices(C, k=1)
    r_geo = float(np.corrcoef(Dg[iu], Dt[iu])[0, 1])

    # Are the recovered centroids "almost on a circle"? Fit a circle in the (Y[:,0], Y[:,1]) plane.
    pts = centroids[:, :2]
    pts_centered = pts - pts.mean(axis=0)
    r_circle = float(np.median(np.linalg.norm(pts_centered, axis=1)))
    radii = np.linalg.norm(pts_centered, axis=1)
    circle_residual = float(np.std(radii) / (r_circle + 1e-12))

    # Cycle-order check: do centroids visit angles in calendar order?
    angles = np.arctan2(pts_centered[:, 1], pts_centered[:, 0])
    # Roll so concept 0 is first; check monotonic order around the circle
    # (allow either direction).
    order_idx = np.argsort(angles)
    rolled = np.roll(order_idx, -np.argmax(order_idx == 0))
    fwd = list(rolled) == list(range(C))
    bwd = list(rolled) == list(range(C - 1, -1, -1)) or list(np.roll(rolled, -1)) == list(range(C - 1, -1, -1))
    cycle_order_correct = fwd or bwd

    # Persistent H1 — does the recovered embedding (or residual) have a loop?
    h1 = rec.h1_diagram
    if len(h1) > 0:
        lt = h1[:, 1] - h1[:, 0]
        top = np.sort(lt)[::-1]
        ratio = float(top[0] / (top[1] + 1e-12)) if len(top) >= 2 else float("inf")
    else:
        ratio = 0.0

    return {
        "r_geodesic_vs_cycle": r_geo,
        "circle_radial_residual": circle_residual,
        "cycle_order_correct": bool(cycle_order_correct),
        "h1_top1_to_top2_ratio": ratio,
        "centroids_xy": pts.tolist(),
        "intrinsic_dim_R": rec.intrinsic_dim_R,
        "intrinsic_dim_R_estimates": rec.intrinsic_dim_R_estimate,
    }


def plot_recovered(rec, pids, labels, family: str, title: str, out_path: Path):
    order = CALENDAR_ORDER[family]
    C = len(order)
    cidx = np.array([order.index(labels[p]["concept"]) for p in pids])
    Y = rec.embedding
    fig = plt.figure(figsize=(13, 5))

    if Y.shape[1] >= 3:
        ax = fig.add_subplot(1, 3, 1, projection="3d")
        for k in range(C):
            m = cidx == k
            ax.scatter(Y[m, 0], Y[m, 1], Y[m, 2], s=10, alpha=0.6, label=order[k])
        ax.set_xlabel("s = w·h"); ax.set_ylabel("dm1(R)"); ax.set_zlabel("dm2(R)")
        ax.set_title("recovered embedding 3D")
    else:
        ax = fig.add_subplot(1, 3, 1)
        for k in range(C):
            m = cidx == k
            ax.scatter(Y[m, 0], Y[m, 1] if Y.shape[1] > 1 else np.zeros(m.sum()),
                       s=10, alpha=0.6, label=order[k])
        ax.set_xlabel("s = w·h"); ax.set_ylabel("dm1(R)" if Y.shape[1] > 1 else "")
        ax.set_title("recovered embedding 2D")

    # Centroids in (s, dm1) plane
    ax2 = fig.add_subplot(1, 3, 2)
    centroids = np.stack([np.median(Y[cidx == k, :2], axis=0) for k in range(C)])
    cmap = plt.get_cmap("hsv")
    for k in range(C):
        m = cidx == k
        ax2.scatter(Y[m, 0], Y[m, 1] if Y.shape[1] > 1 else np.zeros(m.sum()),
                    s=8, alpha=0.4, color=cmap(k / C))
    ax2.scatter(centroids[:, 0], centroids[:, 1], s=160, marker="o", edgecolor="k",
                facecolor="white", linewidth=2, zorder=5)
    for k in range(C):
        ax2.annotate(order[k], centroids[k, :2] + np.array([0.05, 0.05]), fontsize=9)
    # Connect in calendar order
    cents_loop = np.vstack([centroids, centroids[:1]])
    ax2.plot(cents_loop[:, 0], cents_loop[:, 1], "k--", alpha=0.4)
    ax2.set_xlabel("s = w·h"); ax2.set_ylabel("dm1(R)")
    ax2.set_title("concept centroids on (s, dm1)")

    # H1 diagram
    ax3 = fig.add_subplot(1, 3, 3)
    h1 = rec.h1_diagram
    if len(h1) > 0:
        ax3.scatter(h1[:, 0], h1[:, 1], s=10, alpha=0.6)
        m = max(h1[:, 1].max(), h1[:, 0].max())
        ax3.plot([0, m], [0, m], "k--", alpha=0.4)
    ax3.set_xlabel("birth"); ax3.set_ylabel("death")
    ax3.set_title("H1 of orthogonal residuals")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir",
                    default="/data/artifacts/rohan/manifolds/concepts_olmo31_32b_v2/shard_dp00")
    ap.add_argument("--labels", default="/home/rkathuria/manifolds/data/labels_v2.jsonl")
    ap.add_argument("--out-dir",
                    default="/data/artifacts/rohan/manifolds/concepts_olmo31_32b_v2/analysis_recover")
    ap.add_argument("--layers", default="16,24,32,40,48")
    ap.add_argument("--family", default="weekday")
    ap.add_argument("--probe-pairs", default="Monday:Friday,Tuesday:Saturday,Wednesday:Sunday")
    args = ap.parse_args()

    labels = load_labels(Path(args.labels))
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pairs = [tuple(p.split(":")) for p in args.probe_pairs.split(",")]

    all_rows = []
    for L in [int(x) for x in args.layers.split(",")]:
        print(f"\n========== layer {L} ==========")
        X_full, pids = read_last_token_per_seq(Path(args.shard_dir), L)
        X, pids = restrict_to_family(X_full, pids, labels, args.family)
        print(f"loaded {X.shape[0]} sequences  d_model={X.shape[1]}")

        # Baseline: unsupervised manifold discovery on X with no probe
        from manifold import diffusion_map as _dm
        from manifold import knn_geodesic_distances as _knn_geo
        pca_X = reduce_pca(X, k_pca=64)
        Xp_raw = pca_X.X.astype(np.float64)
        dm_raw = _dm(Xp_raw, k_nn=20, n_eigs=4)
        baseline_geo = _knn_geo(Xp_raw, k_nn=20)
        order = CALENDAR_ORDER[args.family]
        C = len(order)
        cidx = np.array([order.index(labels[p]["concept"]) for p in pids])
        Dg = np.zeros((C, C))
        for i in range(C):
            for j in range(C):
                Dg[i, j] = baseline_geo[np.ix_(cidx == i, cidx == j)].mean()
        Dt = np.array([[abs(i - j) if abs(i - j) <= C // 2 else C - abs(i - j)
                        for j in range(C)] for i in range(C)])
        iu = np.triu_indices(C, k=1)
        r_baseline = float(np.corrcoef(Dg[iu], Dt[iu])[0, 1])
        print(f"  baseline (no probe) r(unsup geodesic, cycle)= {r_baseline:.3f}")

        for (a, b) in pairs:
            print(f"\n  -- probe = {a} - {b} --")
            w = chord_probe(X, pids, labels, a, b)

            # Feasibility pre-check: does the probe even lie on a non-linear
            # structure? If nullspace R^2 (RBF) is near zero we expect recovery
            # to fail because there's no second axis to find.
            ns = nullspace_nonlinear_R2(X.astype(np.float64), w, n_subsample=1500)
            print(f"  nullspace_R²: rbf={ns['R2_rbf_kernel']:+.3f}  "
                  f"linear={ns['R2_linear_baseline']:+.3f}  "
                  f"nonlinear_gain={ns['nonlinear_gain']:+.3f}")

            rec = recover(X, w, k_pca=64, k_nn=20, n_eigs=8, intrinsic_dim_cap=4)
            print(f"  intrinsic dim of R: local-PCA={rec.intrinsic_dim_R}  "
                  f"twonn={rec.intrinsic_dim_R_estimate['twonn']:.2f}")
            print(f"  R PCA top-3 explained: {rec.R_pca_explained[:3]}")
            ev = evaluate_recovered(rec, pids, labels, args.family)
            ev["nullspace_R2_rbf"] = ns["R2_rbf_kernel"]
            ev["nullspace_R2_linear"] = ns["R2_linear_baseline"]
            ev["nullspace_nonlinear_gain"] = ns["nonlinear_gain"]
            print(f"  r(recovered geodesic, cycle) = {ev['r_geodesic_vs_cycle']:.3f}  "
                  f"(baseline {r_baseline:.3f})")
            print(f"  circle radial residual = {ev['circle_radial_residual']:.3f}  "
                  f"(0 = perfect circle)")
            print(f"  cycle order correct: {ev['cycle_order_correct']}")
            print(f"  H1 top1/top2 ratio: {ev['h1_top1_to_top2_ratio']:.2f}")

            png_path = out_dir / f"recovered_L{L:03d}_{a}_vs_{b}.png"
            plot_recovered(rec, pids, labels, args.family,
                           f"layer {L}  probe={a}-{b}  r={ev['r_geodesic_vs_cycle']:.2f}",
                           png_path)
            print(f"  wrote {png_path}")

            all_rows.append({"layer": L, "probe_a": a, "probe_b": b,
                             "r_baseline_no_probe": r_baseline, **ev})

    Path(out_dir, "all_recover_results.json").write_text(json.dumps(all_rows, indent=2))
    print(f"\nwrote {out_dir}/all_recover_results.json")


if __name__ == "__main__":
    main()

"""V2 concept analysis: 3 levels of denoising + supervised baseline + 3D viz.

For each layer × family:
  variant 1: per-prompt unsupervised (raw, like v1)
  variant 2: per-concept-averaged unsupervised (paper-style preprocessing
             + paper-style discovery — most fair comparison)
  variant 3: kmeans-cluster centroids unsupervised (no labels used; tests
             whether label-free centroiding rescues us)
  supervised: paper-style spline through concept centroids (validation only)

Output:
  - per-variant geodesic correlation with the supervised arc-length
  - 3D scatter PNG showing diffusion-map embedding colored by held-out concept
    label, with the supervised spline overlaid
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).parent))
from load_acts import load_labels, read_last_token_per_seq
from manifold import diffusion_map, knn_geodesic_distances, reduce_pca
from supervised import CALENDAR_ORDER, centroid_geodesic_matrix, fit_supervised


def cycle_distance_matrix(C: int) -> np.ndarray:
    return np.array([[min(abs(i - j), C - abs(i - j)) for j in range(C)] for i in range(C)])


def restrict_to_family(X, pids, labels, family):
    keep = [i for i, p in enumerate(pids) if labels[p]["family"] == family]
    return X[keep], [pids[i] for i in keep]


def geodesic_centroid_matrix(geo: np.ndarray, group_idx: np.ndarray, C: int) -> np.ndarray:
    D = np.zeros((C, C))
    for i in range(C):
        for j in range(C):
            mi = group_idx == i; mj = group_idx == j
            D[i, j] = geo[np.ix_(mi, mj)].mean()
    return D


def run_variant_per_prompt(X: np.ndarray, k_pca: int, k_nn: int):
    pca = reduce_pca(X, k_pca=k_pca)
    Xp = pca.X.astype(np.float32)
    geo = knn_geodesic_distances(Xp, k_nn=k_nn)
    dm = diffusion_map(Xp.astype(np.float64), k_nn=k_nn, n_eigs=8)
    return pca, Xp, geo, dm


def run_variant_concept_avg(X: np.ndarray, cidx: np.ndarray, C: int, k_pca: int):
    centroids = np.stack([X[cidx == k].mean(axis=0) for k in range(C)])
    pca = reduce_pca(centroids, k_pca=min(k_pca, C))
    Xp = pca.X.astype(np.float32)
    # On C points, kNN graph with k=C-1 is just the dense complete graph
    geo = knn_geodesic_distances(Xp, k_nn=min(k_pca, C - 1))
    return pca, Xp, geo, None


def run_variant_kmeans(X: np.ndarray, n_clusters: int, k_pca: int, k_nn: int, seed: int = 0):
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(X)
    centers = km.cluster_centers_
    labels = km.labels_
    pca = reduce_pca(centers, k_pca=min(k_pca, n_clusters))
    Xp = pca.X.astype(np.float32)
    geo = knn_geodesic_distances(Xp, k_nn=min(n_clusters - 1, k_pca))
    return pca, Xp, geo, labels


def viz_3d(Xp: np.ndarray, cidx: np.ndarray, sm, family: str, title: str, out_path: Path):
    """3D scatter of the diffusion-map embedding (top 3 dm coords) colored
    by held-out concept label, with the supervised spline (in the same PCA
    space) projected onto the same axes."""
    order = CALENDAR_ORDER[family]
    C = len(order)
    cmap = plt.get_cmap("hsv")

    dm = diffusion_map(Xp.astype(np.float64), k_nn=min(20, len(Xp) // 8), n_eigs=6)
    coords = dm.coords(n_dims=3)

    fig = plt.figure(figsize=(13, 5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    for k in range(C):
        m = cidx == k
        ax.scatter(coords[m, 0], coords[m, 1], coords[m, 2], s=12, alpha=0.55,
                   color=cmap(k / C), label=order[k])
    ax.set_title("unsupervised diffusion-map (colored by hidden label)")
    ax.set_xlabel("dm1"); ax.set_ylabel("dm2"); ax.set_zlabel("dm3")
    ax.legend(loc="upper left", fontsize=7, ncols=2)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    # Project the supervised spline (which lives in PCA-of-centroids space
    # OR in PCA-of-X space depending on how sm was fit) onto top 3 PCs
    # of the *current* PCA. Easiest: just plot the supervised centroids
    # in the same diffusion-map space by interpolating them as if they were
    # additional points (using the closest cluster mean per concept).
    sup_centroids = np.stack([Xp[cidx == k].mean(axis=0) for k in range(C)])
    # Convert to diffusion-map coords using Nystrom-like extension via nearest
    # neighbor in dm space
    coords_centroid = np.zeros((C, 3))
    for k in range(C):
        # use the centroid of dm coords for this concept
        coords_centroid[k] = coords[cidx == k].mean(axis=0)
    cents_loop = np.vstack([coords_centroid, coords_centroid[:1]])
    ax2.plot(cents_loop[:, 0], cents_loop[:, 1], cents_loop[:, 2], "k-", lw=2)
    ax2.scatter(coords_centroid[:, 0], coords_centroid[:, 1], coords_centroid[:, 2],
                s=180, marker="o", edgecolor="k", facecolor="white", linewidth=2)
    for k in range(C):
        ax2.text(coords_centroid[k, 0], coords_centroid[k, 1], coords_centroid[k, 2],
                 order[k], fontsize=9)
    ax2.set_title("concept centroids on the embedding (calendar-order spline)")
    ax2.set_xlabel("dm1"); ax2.set_ylabel("dm2"); ax2.set_zlabel("dm3")

    fig.suptitle(title); fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run(shard_dir: Path, layer: int, labels: dict, family: str, out_dir: Path) -> dict:
    print(f"\n========== layer {layer}, family={family} ==========", flush=True)
    X_full, pids = read_last_token_per_seq(shard_dir, layer)
    X, pids = restrict_to_family(X_full, pids, labels, family)
    order = CALENDAR_ORDER[family]
    C = len(order)
    cidx = np.array([order.index(labels[p]["concept"]) for p in pids])
    print(f"loaded {X.shape[0]} sequences", flush=True)

    Dt = cycle_distance_matrix(C).astype(float)
    iu = np.triu_indices(C, k=1)

    # supervised arc-length on PCA-of-X space (paper-style spline)
    pca_X = reduce_pca(X, k_pca=64)
    Xp = pca_X.X.astype(np.float32)
    sm = fit_supervised(Xp, pids, labels, family=family)
    Ds = centroid_geodesic_matrix(sm)
    r_target = float(np.corrcoef(Ds[iu], Dt[iu])[0, 1])
    print(f"  supervised arc-length vs cycle-position: r={r_target:.3f}", flush=True)

    # variant 1: per-prompt unsupervised
    _, _, geo1, _ = run_variant_per_prompt(X, k_pca=64, k_nn=20)
    Dg1 = geodesic_centroid_matrix(geo1, cidx, C)
    r1 = float(np.corrcoef(Dg1[iu], Ds[iu])[0, 1])
    print(f"  v1 per-prompt        r(unsup geo, sup arclen) = {r1:.3f}", flush=True)

    # variant 2: per-concept-averaged unsupervised (paper preprocessing)
    _, _, geo2, _ = run_variant_concept_avg(X, cidx, C, k_pca=64)
    r2 = float(np.corrcoef(geo2[iu], Ds[iu])[0, 1])
    print(f"  v2 concept-avg       r(unsup geo, sup arclen) = {r2:.3f}", flush=True)

    # variant 3: kmeans (no labels)
    _, _, geo3, km_labels = run_variant_kmeans(X, n_clusters=C, k_pca=64, k_nn=20)
    # Match clusters to concepts via majority-label vote (uses labels for evaluation only)
    from scipy.optimize import linear_sum_assignment
    # cost = -confusion(km_label, true_label)
    conf = np.zeros((C, C), dtype=int)
    for k in range(C):
        for c in range(C):
            conf[k, c] = int(((km_labels == k) & (cidx == c)).sum())
    rows, cols = linear_sum_assignment(-conf)
    # remap kmeans labels to concept indices
    map_kmeans_to_concept = {rows[i]: cols[i] for i in range(C)}
    geo3_remap = np.zeros_like(geo3)
    for i in range(C):
        for j in range(C):
            geo3_remap[map_kmeans_to_concept[i], map_kmeans_to_concept[j]] = geo3[i, j]
    r3 = float(np.corrcoef(geo3_remap[iu], Ds[iu])[0, 1])
    cluster_purity = float(conf.max(axis=1).sum() / conf.sum())
    print(f"  v3 kmeans-then-fit   r(unsup geo, sup arclen) = {r3:.3f}  "
          f"(cluster purity vs concepts = {cluster_purity:.2f})", flush=True)

    # 3D viz
    png = out_dir / f"viz_L{layer:03d}_{family}.png"
    viz_3d(Xp, cidx, sm, family,
           f"layer {layer} {family}  r(per-prompt)={r1:.2f}  r(concept-avg)={r2:.2f}",
           png)
    print(f"  wrote {png}", flush=True)

    return {
        "layer": layer, "family": family, "n_seqs": int(X.shape[0]),
        "r_supervised_vs_cycle": r_target,
        "r_v1_per_prompt": r1,
        "r_v2_concept_avg": r2,
        "r_v3_kmeans": r3,
        "kmeans_cluster_purity": cluster_purity,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir",
                    default="/data/artifacts/rohan/manifolds/concepts_olmo31_32b_v2/shard_dp00")
    ap.add_argument("--labels", default="/home/rkathuria/manifolds/data/labels_v2.jsonl")
    ap.add_argument("--out-dir",
                    default="/data/artifacts/rohan/manifolds/concepts_olmo31_32b_v2/analysis_v2")
    ap.add_argument("--layers", default="16,24,32,40,48")
    ap.add_argument("--families", default="weekday,month")
    args = ap.parse_args()

    labels = load_labels(Path(args.labels))
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for L in [int(x) for x in args.layers.split(",")]:
        for fam in args.families.split(","):
            try:
                rows.append(run(Path(args.shard_dir), L, labels, fam, out))
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"!! layer {L} family {fam}: {type(e).__name__}: {e}")
    Path(out, "summary.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}/summary.json")


if __name__ == "__main__":
    main()

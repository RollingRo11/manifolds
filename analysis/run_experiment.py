"""End-to-end pilot experiment.

Steps (no concept labels used in 1-3):
  1. Load last-token activations from the harvester output for each chosen layer.
  2. Run unsupervised manifold discovery (PCA → diffusion map → tangents → PH).
  3. Compute geodesic distances on the kNN graph.
  4. Validation: load labels, fit supervised centroid spline, compare
     centroid-to-centroid distances on the unsupervised graph vs the
     supervised spline arc lengths. Target r > 0.9.
  5. Probe-tangent test: load any probe directions in /data/artifacts/rohan/santi/probes
     that are at the right d_model, and report tau distribution along the manifold.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from load_acts import load_labels, read_last_token_per_seq
from manifold import (
    diffusion_map,
    estimate_tangents,
    knn_geodesic_distances,
    local_pca_eigengap,
    persistence_diagrams,
    reduce_pca,
    two_nn_intrinsic_dim,
)
from probe_manifold import fit_probe_manifold, report as report_pm
from probe_tangent import evaluate_probe, report_str
from supervised import CALENDAR_ORDER, centroid_geodesic_matrix, fit_supervised


def restrict_to_family(X, pids, labels, family):
    keep_idx = [i for i, p in enumerate(pids) if labels[p]["family"] == family]
    return X[keep_idx], [pids[i] for i in keep_idx]


def run_for_layer(shard_dir, layer, labels, family, out_dir):
    print(f"\n========== layer {layer}, family={family} ==========")
    X_full, pids = read_last_token_per_seq(shard_dir, layer)
    X, pids = restrict_to_family(X_full, pids, labels, family)
    print(f"loaded {X.shape[0]} sequences in family={family}, d_model={X.shape[1]}")
    print(f"act-norm stats: mean={np.linalg.norm(X, axis=1).mean():.2f} "
          f"min={np.linalg.norm(X, axis=1).min():.2f} max={np.linalg.norm(X, axis=1).max():.2f}")

    # 1) PCA reduction
    pca = reduce_pca(X, k_pca=64)
    Xp = pca.X
    cum = np.cumsum(pca.explained)
    print(f"PCA k=64: cumulative explained @ k=2,4,8,16,32 = "
          f"{cum[1]:.3f} {cum[3]:.3f} {cum[7]:.3f} {cum[15]:.3f} {cum[31]:.3f}")

    # 2) Intrinsic dimension
    id_two = two_nn_intrinsic_dim(Xp)
    eigs, id_local = local_pca_eigengap(Xp, k=min(30, X.shape[0] // 4))
    print(f"intrinsic dim: TwoNN={id_two:.2f}  local-PCA-median-eigengap={id_local}")

    # 3) Diffusion map for global coords
    dm = diffusion_map(Xp, k_nn=min(20, X.shape[0] // 8), n_eigs=8)
    print(f"diffusion eigvals (first 6): {dm.eigvals[:6]}")

    # 4) Persistent homology
    res = persistence_diagrams(Xp, max_dim=1)
    h1 = res["dgms"][1]
    if len(h1) > 0:
        lifetimes = h1[:, 1] - h1[:, 0]
        top3 = np.sort(lifetimes)[::-1][:3]
        print(f"H1 (loops): {len(h1)} features  top-3 lifetimes={top3}  ratio top1/top2={(top3[0]/(top3[1]+1e-12)):.2f}")
    else:
        print("H1: no loops")

    # 5) Tangent field — pick intrinsic dim
    intrinsic_dim = max(1, int(round(id_local)))
    intrinsic_dim = min(intrinsic_dim, 4)  # cap for visualisation
    print(f"using intrinsic_dim={intrinsic_dim}")
    tf = estimate_tangents(Xp, intrinsic_dim=intrinsic_dim,
                           k=min(30, X.shape[0] // 4))

    # 6) Geodesic distances on kNN graph
    Dg = knn_geodesic_distances(Xp, k_nn=min(20, X.shape[0] // 8))

    # 7) Supervised validation: per-concept centroids on kNN-geodesic
    sm = fit_supervised(X, pids, labels, family=family)
    sm_pca = fit_supervised(Xp, pids, labels, family=family)  # spline in PCA space

    # Map each sequence to its concept index for centroid-pair distance lookup
    order = CALENDAR_ORDER[family]
    cidx = np.array([order.index(labels[p]["concept"]) for p in pids])
    C = len(order)
    # Mean unsupervised geodesic distance between sequences of concept i and concept j
    Dg_centroid = np.zeros((C, C))
    for i in range(C):
        for j in range(C):
            mask_i = cidx == i
            mask_j = cidx == j
            Dg_centroid[i, j] = Dg[np.ix_(mask_i, mask_j)].mean()

    # Supervised arc-length distances between centroids
    Ds = centroid_geodesic_matrix(sm_pca)
    iu = np.triu_indices(C, k=1)
    r_geo = np.corrcoef(Dg_centroid[iu], Ds[iu])[0, 1]
    print(f"unsupervised geodesic vs supervised arc-length: r = {r_geo:.3f}  (target > 0.9)")

    # Linear distances baseline
    centroid_means = np.stack([X[cidx == k].mean(axis=0) for k in range(C)])
    Dl = np.linalg.norm(centroid_means[:, None] - centroid_means[None, :], axis=-1)
    r_lin = np.corrcoef(Dl[iu], Ds[iu])[0, 1]
    print(f"linear-centroid-distance vs supervised arc-length: r = {r_lin:.3f}  (paper baseline)")

    # 8) Probe-tangent decomposition (manifold-first) AND probe-driven manifold (probe-first)
    probe_dir = Path("/data/artifacts/rohan/santi/probes")
    probe_reports = []
    probe_manifold_reports = []
    if probe_dir.exists():
        # Skip dataset-derived probe files that we shouldn't read in this context.
        SKIP = ("harmbench", "fortress")
        for f in probe_dir.glob("*.npz"):
            if any(s in f.name.lower() for s in SKIP):
                continue
            try:
                npz = np.load(f, allow_pickle=False)
            except Exception:
                continue
            for key in npz.files:
                # Per-layer probes: only use the one matching the current layer (e.g. "L32_*")
                layer_tag = f"L{layer}_"
                if key.startswith("L") and "_" in key and not key.startswith(layer_tag):
                    continue
                arr = npz[key]
                vec = None
                if arr.ndim == 1 and arr.shape[0] == X.shape[1]:
                    vec = arr
                elif arr.ndim == 2 and arr.shape[-1] == X.shape[1]:
                    vec = arr[0]
                if vec is None:
                    continue
                name = f"{f.stem}:{key}" + ("[0]" if arr.ndim == 2 else "")
                rep = evaluate_probe(name, vec, pca, tf)
                probe_reports.append(rep)
                print(report_str(rep))

                # Probe-driven manifold (project probe into pca space, fit mu(s))
                w_pca = pca.components @ vec.astype(np.float32)
                if np.linalg.norm(w_pca) > 1e-6 and Xp.shape[0] >= 80:
                    pm = fit_probe_manifold(Xp.astype(np.float64), w_pca, n_bins=12)
                    probe_manifold_reports.append({
                        "name": name,
                        "curvature_ratio": pm.curvature_ratio,
                        "cos_tangent_to_w_mean": float(pm.cos_tangent_to_w.mean()),
                        "cos_tangent_to_w_min": float(pm.cos_tangent_to_w.min()),
                        "cos_tangent_to_w_max": float(pm.cos_tangent_to_w.max()),
                        "bimodality": pm.bimodality,
                        "score_range": [float(pm.s_bin_centers[0]), float(pm.s_bin_centers[-1])],
                        "score_std": float(np.std(Xp @ (w_pca / np.linalg.norm(w_pca)))),
                    })
                    print(report_pm(pm, name))

    # Persist results
    layer_dir = Path(out_dir) / f"layer_{layer:03d}_{family}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    np.savez(layer_dir / "manifold.npz",
             X=X, X_pca=Xp,
             pca_mean=pca.mean, pca_components=pca.components,
             pca_explained=pca.explained,
             diffusion_eigvals=dm.eigvals, diffusion_eigvecs=dm.eigvecs,
             tangent_basis=tf.tangent_basis, tangent_eigvals=tf.eigvals,
             knn_geodesic=Dg, centroid_geodesic_unsup=Dg_centroid,
             centroid_arclen_sup=Ds, centroid_means=centroid_means,
             cidx=cidx, pids=np.array(pids))
    summary = {
        "layer": layer, "family": family, "n_seqs": int(X.shape[0]),
        "d_model": int(X.shape[1]), "k_pca": 64,
        "pca_cum_explained_at_2_4_8_16_32": [float(cum[1]), float(cum[3]), float(cum[7]),
                                              float(cum[15]), float(cum[31])],
        "twonn_intrinsic_dim": float(id_two),
        "local_pca_intrinsic_dim": int(round(id_local)),
        "intrinsic_dim_used": intrinsic_dim,
        "h1_top3_lifetimes": top3.tolist() if len(h1) > 0 else [],
        "h1_top1_to_top2_ratio": float(top3[0] / (top3[1] + 1e-12)) if len(h1) >= 2 else None,
        "r_unsup_geodesic_vs_sup_arclen": float(r_geo),
        "r_lin_centroid_vs_sup_arclen": float(r_lin),
        "probe_reports": [
            {"name": r.name, "tau_mean": r.tau_mean, "tau_median": r.tau_median,
             "random_baseline": r.random_baseline,
             "fraction_explained_by_pca": r.fraction_explained_by_pca,
             "angle_to_M_deg_mean": r.angle_to_manifold_deg_mean}
            for r in probe_reports
        ],
        "probe_manifold_reports": probe_manifold_reports,
    }
    (layer_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {layer_dir}/{{manifold.npz, summary.json}}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", default="/data/artifacts/rohan/manifolds/concepts_olmo31_32b_v1")
    ap.add_argument("--labels", default="/home/rkathuria/manifolds/data/labels.jsonl")
    ap.add_argument("--out-dir", default="/data/artifacts/rohan/manifolds/concepts_olmo31_32b_v1/analysis")
    ap.add_argument("--layers", default="16,24,32,40,48,56")
    ap.add_argument("--families", default="weekday,month")
    args = ap.parse_args()

    labels = load_labels(Path(args.labels))
    layers = [int(x) for x in args.layers.split(",")]
    families = args.families.split(",")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    all_summaries = []
    for L in layers:
        for fam in families:
            try:
                s = run_for_layer(Path(args.shard_dir), L, labels, fam, args.out_dir)
                all_summaries.append(s)
            except Exception as e:
                print(f"!! layer {L} family {fam}: {type(e).__name__}: {e}")
    Path(args.out_dir, "all_summaries.json").write_text(json.dumps(all_summaries, indent=2))
    print(f"\n=== wrote {args.out_dir}/all_summaries.json ===")


if __name__ == "__main__":
    main()

"""Probe-driven manifold analysis on the eval-awareness harvest.

For each layer, for each compatible probe (especially eval-awareness-related):
  - project the probe direction into the activation PCA space
  - fit mu(s) = E[h | w.h ~ s] (probe_manifold.fit_probe_manifold)
  - report curvature ratio, tangent drift, score-density bimodality
  - secondary: compare the probe's induced ordering to the held-back frame_score

Also does unsupervised manifold discovery on the same data (for context).
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
    local_pca_eigengap,
    persistence_diagrams,
    reduce_pca,
    two_nn_intrinsic_dim,
)
from probe_manifold import fit_probe_manifold, random_curvature_baseline, report as report_pm
from probe_tangent import evaluate_probe, report_str

EVAL_PROBE_KEYWORDS = ("testing_vs_conversation", "hua_et_al")


def run_for_layer(shard_dir: Path, layer: int, labels: dict[int, dict], out_dir: Path):
    print(f"\n========== layer {layer} ==========")
    X, pids = read_last_token_per_seq(shard_dir, layer)
    print(f"loaded {X.shape[0]} sequences  d_model={X.shape[1]}")

    pca = reduce_pca(X, k_pca=64)
    Xp = pca.X.astype(np.float64)
    cum = np.cumsum(pca.explained)
    print(f"PCA cum-explained @ k=2,4,8,16,32 = "
          f"{cum[1]:.3f} {cum[3]:.3f} {cum[7]:.3f} {cum[15]:.3f} {cum[31]:.3f}")

    id_two = two_nn_intrinsic_dim(Xp)
    _, id_local = local_pca_eigengap(Xp, k=30)
    print(f"intrinsic dim: TwoNN={id_two:.2f}  local-PCA-median-eigengap={id_local}")

    dm = diffusion_map(Xp, k_nn=20, n_eigs=8)
    print(f"diffusion eigvals (first 6): {dm.eigvals[:6]}")

    res = persistence_diagrams(Xp, max_dim=1)
    h1 = res["dgms"][1]
    if len(h1) > 0:
        lt = h1[:, 1] - h1[:, 0]; top3 = np.sort(lt)[::-1][:3]
        print(f"H1 top-3 lifetimes: {top3}  ratio top1/top2={(top3[0]/(top3[1]+1e-12)):.2f}")

    intrinsic_dim = max(1, min(int(round(id_local)), 4))
    tf = estimate_tangents(Xp.astype(np.float32), intrinsic_dim=intrinsic_dim, k=30)

    # Randomization baseline for curvature ratio: average over many random
    # directions in the same PCA subspace. Anything above this distribution
    # is genuinely curved beyond finite-sample noise.
    rand_mu, rand_sd = random_curvature_baseline(Xp, n_dirs=64, n_bins=12, seed=0)
    print(f"random-direction curvature baseline: mu={rand_mu:.3f}  sd={rand_sd:.3f}")

    # Frame score: held-back label, only used post-hoc
    fs = np.array([labels[p]["frame_score"] for p in pids])

    # Per-probe analysis
    probe_dir = Path("/data/artifacts/rohan/santi/probes")
    SKIP = ("harmbench", "fortress")
    layer_tag = f"L{layer}_"

    pm_reports, tau_reports = [], []
    for f in probe_dir.glob("*.npz"):
        if any(s in f.name.lower() for s in SKIP):
            continue
        try:
            npz = np.load(f, allow_pickle=False)
        except Exception:
            continue
        for key in npz.files:
            arr = npz[key]
            if key.startswith("L") and "_" in key and not key.startswith(layer_tag):
                continue
            vec = None
            if arr.ndim == 1 and arr.shape[0] == X.shape[1]:
                vec = arr
            elif arr.ndim == 2 and arr.shape[-1] == X.shape[1]:
                vec = arr[0]
            if vec is None:
                continue
            name = f"{f.stem}:{key}" + ("[0]" if arr.ndim == 2 else "")

            # Tangent fraction along the unsupervised manifold
            tau = evaluate_probe(name, vec.astype(np.float32), pca, tf)
            tau_reports.append(tau)

            # Probe-driven manifold
            w_pca = pca.components @ vec.astype(np.float32)
            pm = fit_probe_manifold(Xp, w_pca, n_bins=12)

            # Probe-score ordering vs held-back frame_score
            s_proj = X @ vec.astype(np.float32)
            r_score_frame = float(np.corrcoef(s_proj, fs)[0, 1])
            spearman = float(np.corrcoef(np.argsort(np.argsort(s_proj)),
                                          np.argsort(np.argsort(fs)))[0, 1])

            is_eval = any(kw in f.stem for kw in EVAL_PROBE_KEYWORDS)
            tag = "[EVAL-AWARENESS]" if is_eval else ""
            print(f"\n--- probe {name} {tag} ---")
            print(report_str(tau))
            print(report_pm(pm, name))
            print(f"  s = w.h vs frame_score (0..6): pearson={r_score_frame:.3f}  spearman={spearman:.3f}")

            curv_z = (pm.curvature_ratio - rand_mu) / (rand_sd + 1e-9)
            pm_reports.append({
                "name": name, "is_eval_aware": is_eval,
                "tau_mean": tau.tau_mean, "tau_random_baseline": tau.random_baseline,
                "fraction_explained_by_pca": tau.fraction_explained_by_pca,
                "curvature_ratio": pm.curvature_ratio,
                "curvature_random_baseline_mu": rand_mu,
                "curvature_random_baseline_sd": rand_sd,
                "curvature_z_vs_random": curv_z,
                "cos_tangent_to_w_mean": float(pm.cos_tangent_to_w.mean()),
                "cos_tangent_to_w_min": float(pm.cos_tangent_to_w.min()),
                "cos_tangent_to_w_max": float(pm.cos_tangent_to_w.max()),
                "bimodality": pm.bimodality,
                "score_pearson_vs_frame_score": r_score_frame,
                "score_spearman_vs_frame_score": spearman,
                "score_range": [float(pm.s_bin_centers[0]), float(pm.s_bin_centers[-1])],
            })
            print(f"  curvature z-score vs random: {curv_z:+.2f}σ  "
                  f"(probe ratio {pm.curvature_ratio:.3f}, random {rand_mu:.3f}±{rand_sd:.3f})")

    layer_dir = out_dir / f"layer_{layer:03d}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    np.savez(layer_dir / "manifold.npz",
             X=X, X_pca=pca.X,
             pca_mean=pca.mean, pca_components=pca.components, pca_explained=pca.explained,
             diffusion_eigvals=dm.eigvals, diffusion_eigvecs=dm.eigvecs,
             tangent_basis=tf.tangent_basis, tangent_eigvals=tf.eigvals,
             frame_score=fs, pids=np.array(pids))
    summary = {
        "layer": layer, "n_seqs": int(X.shape[0]), "d_model": int(X.shape[1]),
        "pca_cum_explained_at_2_4_8_16_32": [float(cum[i]) for i in (1, 3, 7, 15, 31)],
        "twonn_intrinsic_dim": float(id_two),
        "local_pca_intrinsic_dim": int(round(id_local)),
        "intrinsic_dim_used": intrinsic_dim,
        "h1_top3_lifetimes": top3.tolist() if len(h1) else [],
        "probe_reports": pm_reports,
    }
    (layer_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir",
                    default="/data/artifacts/rohan/manifolds/eval_awareness_olmo31_32b_v1/shard_dp00")
    ap.add_argument("--labels", default="/home/rkathuria/manifolds/data/eval_awareness_labels.jsonl")
    ap.add_argument("--out-dir",
                    default="/data/artifacts/rohan/manifolds/eval_awareness_olmo31_32b_v1/analysis")
    ap.add_argument("--layers", default="16,24,32,40,48")
    args = ap.parse_args()
    labels = load_labels(Path(args.labels))
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for L in [int(x) for x in args.layers.split(",")]:
        try:
            summaries.append(run_for_layer(Path(args.shard_dir), L, labels, out_dir))
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"!! layer {L}: {type(e).__name__}: {e}")
    Path(out_dir, "all_summaries.json").write_text(json.dumps(summaries, indent=2))
    print(f"\n=== wrote {out_dir}/all_summaries.json ===")


if __name__ == "__main__":
    main()

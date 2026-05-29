"""Evaluate SAVE on the paper's other manifold types: temperature (line),
age (line), years (helix), plus the days/circle from v3.

For each manifold:
  - Build a probe = diff-of-means between two extremes of the value range
  - Run SAVE
  - Compute Procrustes between SAVE-recovered centroids and supervised PCA
    centroids (one centroid per discrete value)
  - Compare to:
      v1 (per-bin SIR)
      v3 (global residual PCA, "passenger probe")
      Procrustes from random probes (10 trials)
  - Plot the SAVE 3D embedding colored by the value, alongside supervised PCA
"""
from __future__ import annotations

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
from save_recovery import save_recovery
from sir_recovery import slice_and_local_pca
from sir_recovery_v3 import global_residual_recover


def restrict_to_family(X, pids, labels, family):
    keep = [i for i, p in enumerate(pids) if labels[p]["family"] == family]
    return X[keep], [pids[i] for i in keep]


def per_value_centroids(emb, values):
    """For each unique value, compute the mean of activations with that value."""
    uniq = sorted(set(values))
    cents = np.stack([emb[np.array(values) == v].mean(axis=0) for v in uniq])
    return uniq, cents


def chord_probe_extremes(X, values, q_low=0.2, q_high=0.8):
    """Build a probe as diff-of-means between low and high extremes of value."""
    arr = np.array(values, dtype=np.float64)
    lo = np.quantile(arr, q_low)
    hi = np.quantile(arr, q_high)
    low_idx = arr <= lo
    high_idx = arr >= hi
    w = X[high_idx].mean(axis=0) - X[low_idx].mean(axis=0)
    return w / (np.linalg.norm(w) + 1e-12)


def evaluate_one(family: str, X: np.ndarray, values: list, layer: int) -> dict:
    print(f"\n========== {family} (layer {layer}, n={X.shape[0]}) ==========")
    print(f"  unique values: {len(set(values))}  range: [{min(values)}, {max(values)}]")

    # --- Supervised baseline ---
    sup_emb = PCA(n_components=3).fit_transform(X)
    uniq, sup_cents = per_value_centroids(sup_emb, values)

    # --- Probe: diff-of-means between extremes ---
    w = chord_probe_extremes(X, values)

    # --- SAVE ---
    sv = save_recovery(X, w, n_slices=10, d_M=2)
    _, save_cents = per_value_centroids(sv.embedding, values)
    _, _, save_proc = procrustes(sup_cents, save_cents)
    print(f"  SAVE: procrustes={save_proc:.4f}  top-5 eigvals={[round(x,2) for x in sv.eigvals[:5]]}")

    # --- v1 (with min_per_bin lowered for small datasets) ---
    n_bins_v1 = max(5, min(20, X.shape[0] // 6))
    try:
        v1 = slice_and_local_pca(X, w, n_bins=n_bins_v1, n_partner=2, min_per_bin=3)
        _, v1_cents = per_value_centroids(v1.embedding, values)
        _, _, v1_proc = procrustes(sup_cents, v1_cents)
    except (ValueError, TypeError):
        v1_proc = float("nan")
    print(f"  v1:   procrustes={v1_proc:.4f}  (n_bins={n_bins_v1})")

    # --- v3 ---
    v3 = global_residual_recover(X, w, n_partner=2)
    _, v3_cents = per_value_centroids(v3, values)
    _, _, v3_proc = procrustes(sup_cents, v3_cents)
    print(f"  v3:   procrustes={v3_proc:.4f}")

    # --- random × 10 baseline (only SAVE since that's the recommended algorithm) ---
    rng = np.random.default_rng(0)
    rand_procs, rand_top_eigs = [], []
    for _ in range(10):
        wr = rng.standard_normal(X.shape[1]); wr /= np.linalg.norm(wr)
        sv_r = save_recovery(X, wr, n_slices=10, d_M=2)
        _, rc = per_value_centroids(sv_r.embedding, values)
        _, _, rp = procrustes(sup_cents, rc)
        rand_procs.append(rp)
        rand_top_eigs.append(sv_r.eigvals[0])
    print(f"  SAVE random×10: procrustes={np.mean(rand_procs):.4f}±{np.std(rand_procs):.4f}  "
          f"top-eig={np.mean(rand_top_eigs):.2f}±{np.std(rand_top_eigs):.2f}")

    return {
        "family": family, "layer": layer, "n": int(X.shape[0]),
        "n_unique_values": int(len(set(values))),
        "save_procrustes": float(save_proc),
        "v1_procrustes": float(v1_proc),
        "v3_procrustes": float(v3_proc),
        "save_top5_eigvals": [float(x) for x in sv.eigvals[:5]],
        "save_random_procrustes_mean": float(np.mean(rand_procs)),
        "save_random_procrustes_std": float(np.std(rand_procs)),
        "save_random_top_eig_mean": float(np.mean(rand_top_eigs)),
        "informative_top_eig": float(sv.eigvals[0]),
        "spectral_gap_ratio": float(sv.eigvals[0] / np.mean(rand_top_eigs)),
        "sup_cents": sup_cents.tolist(),
        "save_cents": save_cents.tolist(),
        "values": list(uniq),
    }


def plot_recovery(result: dict, out_path: Path):
    family = result["family"]
    sup = np.array(result["sup_cents"])
    rec = np.array(result["save_cents"])
    values = result["values"]
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(min(values), max(values))

    fig = plt.figure(figsize=(13, 6))
    ax_l = fig.add_subplot(1, 2, 1, projection="3d")
    sc = ax_l.scatter(sup[:, 0], sup[:, 1], sup[:, 2],
                       c=values, cmap=cmap, norm=norm, s=30)
    ax_l.plot(sup[:, 0], sup[:, 1], sup[:, 2], "k-", alpha=0.3, lw=0.8)
    ax_l.set_title(f"supervised PCA (truth)\n{family}, n_unique={len(values)}")

    ax_r = fig.add_subplot(1, 2, 2, projection="3d")
    sc = ax_r.scatter(rec[:, 0], rec[:, 1], rec[:, 2],
                       c=values, cmap=cmap, norm=norm, s=30)
    ax_r.plot(rec[:, 0], rec[:, 1], rec[:, 2], "k-", alpha=0.3, lw=0.8)
    ax_r.set_title(f"SAVE recovered\nprocrustes={result['save_procrustes']:.3f}  "
                   f"spectral-gap={result['spectral_gap_ratio']:.2f}×")
    fig.colorbar(sc, ax=[ax_l, ax_r], shrink=0.7, label=f"{family} value")
    fig.suptitle(f"SAVE on {family} manifold  (layer {result['layer']})")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    layer = 16
    labels_v4 = load_labels(Path("/home/rkathuria/manifolds/data/labels_v4_paper.jsonl"))
    X4_full, pids4 = read_last_token_per_seq(
        Path("/data/artifacts/rohan/manifolds/paper_olmo31_32b_v4/shard_dp00"), layer)

    out_dir = Path("/home/rkathuria/manifolds/figures/paper_manifolds")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for fam in ["temperature", "age", "year"]:
        X, pids = restrict_to_family(X4_full, pids4, labels_v4, fam)
        values = [int(labels_v4[p]["value"]) for p in pids]
        res = evaluate_one(fam, X, values, layer)
        plot_recovery(res, out_dir / f"L{layer:03d}_{fam}.png")
        all_results.append(res)

    # Also rerun on v3 weekday data (circle) for completeness
    labels_v3 = load_labels(Path("/home/rkathuria/manifolds/data/labels_v3.jsonl"))
    X3_full, pids3 = read_last_token_per_seq(
        Path("/data/artifacts/rohan/manifolds/concepts_olmo31_32b_v3/shard_dp00"), layer)
    Xw, pidsw = restrict_to_family(X3_full, pids3, labels_v3, "weekday")
    # Map weekday concept to value 0..6 for plotting
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    valuesw = [order.index(labels_v3[p]["concept"]) for p in pidsw]
    res = evaluate_one("days", Xw, valuesw, layer)
    plot_recovery(res, out_dir / f"L{layer:03d}_days.png")
    all_results.append(res)

    Path(out_dir, "summary.json").write_text(json.dumps(all_results, indent=2))
    print("\n=== SUMMARY ===")
    print(f"{'family':>14} {'n':>6} {'sup-proc':>10} {'SAVE-proc':>10} {'rand-proc':>14} "
          f"{'spectral-gap':>14}")
    for r in all_results:
        rand = f"{r['save_random_procrustes_mean']:.3f}±{r['save_random_procrustes_std']:.3f}"
        print(f"{r['family']:>14} {r['n']:>6} {0.0:>10.3f} {r['save_procrustes']:>10.3f} "
              f"{rand:>14} {r['spectral_gap_ratio']:>13.2f}×")


if __name__ == "__main__":
    main()

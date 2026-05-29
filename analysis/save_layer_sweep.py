"""SAVE layer-sweep on paper manifolds — temperature, age, year, days.

For each layer × manifold:
  - Run SAVE with chord probe (high-extreme − low-extreme)
  - Compute Procrustes vs supervised PCA centroids
  - Compute random-probe baseline (× 5 trials) for spectral gap

Output: line plots showing how SAVE quality varies with layer depth.
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


def chord_probe_extremes(X, values, q_low=0.2, q_high=0.8):
    arr = np.array(values, dtype=np.float64)
    lo = np.quantile(arr, q_low); hi = np.quantile(arr, q_high)
    w = X[arr >= hi].mean(axis=0) - X[arr <= lo].mean(axis=0)
    return w / (np.linalg.norm(w) + 1e-12)


def per_value_centroids(emb, values):
    uniq = sorted(set(values))
    return np.stack([emb[np.array(values) == v].mean(axis=0) for v in uniq])


def evaluate_one(X, values, layer):
    """Run SAVE + random baseline; return summary dict."""
    sup_emb = PCA(n_components=3).fit_transform(X)
    sup_cents = per_value_centroids(sup_emb, values)

    w_pos = chord_probe_extremes(X, values)
    sv = save_recovery(X, w_pos, n_slices=10, d_M=2)
    save_cents = per_value_centroids(sv.embedding, values)
    _, _, save_proc = procrustes(sup_cents, save_cents)

    rng = np.random.default_rng(layer)  # different seed per layer for reproducibility
    rand_top_eigs = []
    rand_procs = []
    for _ in range(5):
        wr = rng.standard_normal(X.shape[1]); wr /= np.linalg.norm(wr)
        sv_r = save_recovery(X, wr, n_slices=10, d_M=2)
        rand_cents = per_value_centroids(sv_r.embedding, values)
        _, _, p = procrustes(sup_cents, rand_cents)
        rand_top_eigs.append(sv_r.eigvals[0])
        rand_procs.append(p)

    return {
        "layer": layer,
        "save_procrustes": float(save_proc),
        "save_top_eig": float(sv.eigvals[0]),
        "save_top5_eigvals": [float(x) for x in sv.eigvals[:5]],
        "rand_procrustes_mean": float(np.mean(rand_procs)),
        "rand_procrustes_std": float(np.std(rand_procs)),
        "rand_top_eig_mean": float(np.mean(rand_top_eigs)),
        "spectral_gap_ratio": float(sv.eigvals[0] / np.mean(rand_top_eigs)),
    }


def main():
    layers = [16, 24, 32, 40, 48]
    results = {}

    # Paper-faithful manifolds
    labels_v4 = load_labels(Path("/home/rkathuria/manifolds/data/labels_v4_paper.jsonl"))
    for L in layers:
        X_full, pids = read_last_token_per_seq(
            Path("/data/artifacts/rohan/manifolds/paper_olmo31_32b_v4/shard_dp00"), L)
        for fam in ["temperature", "age", "year"]:
            keep = [i for i, p in enumerate(pids) if labels_v4[p]["family"] == fam]
            X = X_full[keep]
            values = [int(labels_v4[pids[i]]["value"]) for i in keep]
            res = evaluate_one(X, values, L)
            results.setdefault(fam, []).append(res)
            print(f"[L{L:02d} {fam:>11}] save_proc={res['save_procrustes']:.4f}  "
                  f"rand_proc={res['rand_procrustes_mean']:.4f}  "
                  f"gap={res['spectral_gap_ratio']:.2f}×")

    # Days manifold (from v3 weekday shard)
    labels_v3 = load_labels(Path("/home/rkathuria/manifolds/data/labels_v3.jsonl"))
    weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    for L in layers:
        X_full, pids = read_last_token_per_seq(
            Path("/data/artifacts/rohan/manifolds/concepts_olmo31_32b_v3/shard_dp00"), L)
        keep = [i for i, p in enumerate(pids) if labels_v3[p]["family"] == "weekday"]
        X = X_full[keep]
        values = [weekday_order.index(labels_v3[pids[i]]["concept"]) for i in keep]
        res = evaluate_one(X, values, L)
        results.setdefault("days", []).append(res)
        print(f"[L{L:02d} {'days':>11}] save_proc={res['save_procrustes']:.4f}  "
              f"rand_proc={res['rand_procrustes_mean']:.4f}  "
              f"gap={res['spectral_gap_ratio']:.2f}×")

    Path("/home/rkathuria/manifolds/figures/save_layer_sweep.json").write_text(
        json.dumps(results, indent=2))

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for fam, color in zip(["temperature", "age", "year", "days"],
                            ["C0", "C1", "C2", "C3"]):
        rs = results[fam]
        ls = [r["layer"] for r in rs]
        save_procs = [r["save_procrustes"] for r in rs]
        rand_procs = [r["rand_procrustes_mean"] for r in rs]
        rand_stds = [r["rand_procrustes_std"] for r in rs]
        gaps = [r["spectral_gap_ratio"] for r in rs]
        axes[0].plot(ls, save_procs, "o-", color=color, label=f"{fam} (SAVE)")
        axes[0].errorbar(ls, rand_procs, yerr=rand_stds, fmt="x:", color=color, alpha=0.6,
                          label=f"{fam} (random×5)")
        axes[1].plot(ls, gaps, "o-", color=color, label=fam)

    axes[0].set_xlabel("layer"); axes[0].set_ylabel("Procrustes disparity (lower = better)")
    axes[0].set_title("SAVE recovery quality vs layer")
    axes[0].legend(loc="best", fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("layer"); axes[1].set_ylabel("spectral gap (informative / random top eig)")
    axes[1].set_title("Probe-faithfulness vs layer (>1 means probe earns its keep)")
    axes[1].axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    axes[1].legend(loc="best", fontsize=8); axes[1].grid(alpha=0.3)

    fig.suptitle("SAVE layer sweep across paper manifolds — OLMo-3.1-32B-Think")
    fig.tight_layout()
    out = Path("/home/rkathuria/manifolds/figures/save_layer_sweep.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nwrote {out}")

    # --- Best layer per manifold ---
    print("\n=== best layer per manifold (lowest Procrustes) ===")
    for fam, rs in results.items():
        best = min(rs, key=lambda r: r["save_procrustes"])
        print(f"  {fam:>12}: layer {best['layer']}  proc={best['save_procrustes']:.4f}  "
              f"gap={best['spectral_gap_ratio']:.2f}×")
    print("\n=== best layer per manifold (highest spectral gap) ===")
    for fam, rs in results.items():
        best = max(rs, key=lambda r: r["spectral_gap_ratio"])
        print(f"  {fam:>12}: layer {best['layer']}  gap={best['spectral_gap_ratio']:.2f}×  "
              f"proc={best['save_procrustes']:.4f}")


if __name__ == "__main__":
    main()

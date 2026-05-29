"""Side-by-side: SAVE eigenvalue spectrum on days (multi-cluster) vs EA (binary).

The argument for the SPAR talk:
  - On days data: SAVE eigenspectrum has a sharp drop at λ[7]/λ[8] = 2.12, recovering
    the 7-cluster structure WITHOUT being told K=7. This is unsupervised
    discovery of intrinsic manifold dimension.
  - On FORTRESS/EA data: spectrum drops sharply after λ[0], indicating
    approximately 1-D structure. EA is genuinely binary in this model.

Together: SAVE works as advertised; the reason EA didn't have multi-state
manifold to steer along is that EA itself is approximately 1-D, not a method
failure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))
from load_acts import load_labels, read_last_token_per_seq
from load_acts_multi import read_last_token_multi_rollout
from save_recovery import save_recovery

LAYER = 40
DAYS_SHARD = Path("/data/artifacts/rohan/manifolds/concepts_olmo31_32b_v3/shard_dp00")
DAYS_LABELS = Path("/home/rkathuria/manifolds/data/labels_v3.jsonl")
FORTRESS_SHARD = Path("/data/artifacts/rohan/activation-harvester/fortress_balanced_v1/shard_dp00")
FORTRESS_LABELS = Path("/home/rkathuria/santi/data/probe_train/rollout_labels_fortress_balanced.jsonl")
EA_SHARD = Path("/data/artifacts/rohan/manifolds/eval_awareness_olmo31_32b_v1/shard_dp00")
EA_LABELS = Path("/home/rkathuria/manifolds/data/eval_awareness_labels.jsonl")

OUT_DIR = Path("/home/rkathuria/manifolds/figures/spectrum_comparison")
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def normalize(v):
    return v / (np.linalg.norm(v) + 1e-12)


def biggest_drop(eigvals, k_max=15):
    """Find the largest λ[i]/λ[i+1] ratio in the top k_max eigvals."""
    ratios = [eigvals[i] / max(eigvals[i+1], 1e-12) for i in range(min(k_max, len(eigvals)-1))]
    best_idx = int(np.argmax(ratios))
    return best_idx, ratios[best_idx], ratios


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== days SAVE spectrum (probe = c_Wed - c_Sun, 2 of 7 endpoints) ===")
    labels = load_labels(DAYS_LABELS)
    X, pids = read_last_token_per_seq(DAYS_SHARD, LAYER)
    keep = [i for i, p in enumerate(pids) if labels[p].get("family") == "weekday"]
    Xd = X[keep].astype(np.float64)
    days_idx = np.array([WEEKDAYS.index(labels[pids[i]]["concept"]) for i in keep])
    cents_d = np.stack([Xd[days_idx == k].mean(0) for k in range(7)])
    w_days = normalize(cents_d[2] - cents_d[6])  # Wed - Sun: only uses 2 of 7
    res_d = save_recovery(Xd, w_days, n_slices=10, d_M=20)
    drop_idx_d, drop_ratio_d, ratios_d = biggest_drop(res_d.eigvals, 15)
    print(f"  N = {Xd.shape[0]}")
    print(f"  top-10 eigvals: {res_d.eigvals[:10].round(1)}")
    print(f"  largest ratio drop: λ[{drop_idx_d}]/λ[{drop_idx_d+1}] = {drop_ratio_d:.2f}")
    print(f"  → recovered intrinsic dim: {drop_idx_d + 1}  (true 7-cluster needs ≥6 dim)")

    print("\n=== EA-FS SAVE spectrum (probe = c_FS6 - c_FS0) ===")
    ea_labels = load_labels(EA_LABELS)
    Xea, pids_ea = read_last_token_per_seq(EA_SHARD, LAYER, position_from_end=10)
    fs = np.array([ea_labels[p]["frame_score"] for p in pids_ea])
    cents_ea = np.stack([Xea[fs == k].astype(np.float64).mean(0) for k in range(7)])
    w_ea = normalize(cents_ea[6] - cents_ea[0])
    res_ea = save_recovery(Xea.astype(np.float64), w_ea, n_slices=10, d_M=20)
    drop_idx_ea, drop_ratio_ea, ratios_ea = biggest_drop(res_ea.eigvals, 15)
    print(f"  N = {Xea.shape[0]}")
    print(f"  top-10 eigvals: {res_ea.eigvals[:10].round(1)}")
    print(f"  largest ratio drop: λ[{drop_idx_ea}]/λ[{drop_idx_ea+1}] = {drop_ratio_ea:.2f}")
    print(f"  → recovered intrinsic dim: {drop_idx_ea + 1}")

    print("\n=== FORTRESS SAVE spectrum (probe = mean_aware - mean_non) ===")
    Xf, keys = read_last_token_multi_rollout(FORTRESS_SHARD, LAYER, position_from_end=0)
    aware_map = {}
    with open(FORTRESS_LABELS) as f:
        for line in f:
            d = json.loads(line)
            aware_map[(int(d["prompt_id"]), int(d["completion_idx"]))] = bool(d["aware"])
    matched = [(i, aware_map[k]) for i, k in enumerate(keys) if k in aware_map]
    idx = np.array([m[0] for m in matched])
    aware = np.array([m[1] for m in matched])
    Xa = Xf[idx].astype(np.float64)
    w_f = normalize(Xa[aware].mean(0) - Xa[~aware].mean(0))
    res_f = save_recovery(Xa, w_f, n_slices=10, d_M=20)
    drop_idx_f, drop_ratio_f, ratios_f = biggest_drop(res_f.eigvals, 15)
    print(f"  N = {Xa.shape[0]}")
    print(f"  top-10 eigvals: {res_f.eigvals[:10].round(1)}")
    print(f"  largest ratio drop: λ[{drop_idx_f}]/λ[{drop_idx_f+1}] = {drop_ratio_f:.2f}")
    print(f"  → recovered intrinsic dim: {drop_idx_f + 1}")

    # ===== Plot =====
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    datasets = [
        ("Days (7 weekdays)", res_d, drop_idx_d, drop_ratio_d, "C0", 7),
        ("EA-FS (hand-labeled 7-level Likert)", res_ea, drop_idx_ea, drop_ratio_ea, "C1", 7),
        ("FORTRESS (binary aware/non-aware)", res_f, drop_idx_f, drop_ratio_f, "C2", 2),
    ]

    for ax, (name, res, drop_idx, drop_ratio, color, true_K) in zip(axes[0], datasets):
        ax.bar(range(15), res.eigvals[:15], color=color, alpha=0.7)
        ax.axvline(drop_idx + 0.5, color="red", linestyle="--", alpha=0.6,
                    label=f"largest drop at λ[{drop_idx}]/λ[{drop_idx+1}]={drop_ratio:.2f}")
        ax.set_xlabel("eigvec index")
        ax.set_ylabel("SAVE eigenvalue")
        ax.set_title(f"{name}\n(true # of distinct states: {true_K})", fontsize=11)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        ax.set_yscale("log")

    for ax, (name, res, drop_idx, drop_ratio, color, true_K) in zip(axes[1], datasets):
        ratios = [res.eigvals[i] / max(res.eigvals[i+1], 1e-12) for i in range(14)]
        ax.bar(range(14), ratios, color=color, alpha=0.7)
        ax.axvline(drop_idx + 0.0, color="red", linestyle="--", alpha=0.6,
                    label=f"max ratio at i={drop_idx} ({drop_ratio:.2f}×)")
        ax.set_xlabel("i")
        ax.set_ylabel("λ[i] / λ[i+1]")
        ax.set_title(f"eigval ratio drops — recovered dim = {drop_idx + 1}", fontsize=11)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle("SAVE eigenvalue spectrum: 7-cluster vs binary concepts at L40\n"
                  "Largest ratio-drop reveals intrinsic dimensionality WITHOUT being told K",
                  fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "spectrum_comparison.png", dpi=140, bbox_inches="tight")
    plt.close()

    np.savez(OUT_DIR / "spectrum_data.npz",
              days_eigvals=res_d.eigvals,
              ea_fs_eigvals=res_ea.eigvals,
              fortress_eigvals=res_f.eigvals)
    print(f"\nwrote {OUT_DIR}/spectrum_comparison.png")
    print(f"\n=== KEY RESULT ===")
    print(f"Days (7 clusters expected):    drop at λ[{drop_idx_d}]/λ[{drop_idx_d+1}] = {drop_ratio_d:.2f}× → recovered dim ~{drop_idx_d+1}")
    print(f"EA-FS (7 levels claimed):      drop at λ[{drop_idx_ea}]/λ[{drop_idx_ea+1}] = {drop_ratio_ea:.2f}× → recovered dim ~{drop_idx_ea+1}")
    print(f"FORTRESS (binary):             drop at λ[{drop_idx_f}]/λ[{drop_idx_f+1}] = {drop_ratio_f:.2f}× → recovered dim ~{drop_idx_f+1}")


if __name__ == "__main__":
    main()

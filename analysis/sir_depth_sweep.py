"""SIR benchmark across layers — does cycle recovery depend on depth?"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def run_one(layer: int, base_dir: Path) -> dict:
    out_json = base_dir / f"sir_L{layer:03d}.json"
    out_png = base_dir / f"sir_L{layer:03d}.png"
    cmd = [
        "uv", "run", "python", "analysis/run_sir_benchmark.py",
        "--layer", str(layer),
        "--n-bins", "20", "--n-partner", "2",
        "--out-png", str(out_png),
        "--out-json", str(out_json),
    ]
    print(f"\n=== layer {layer} ===", flush=True)
    subprocess.run(cmd, check=True, cwd="/home/rkathuria/manifolds",
                   env={"PYTHONUNBUFFERED": "1", "PATH": "/usr/bin:/bin:/home/rkathuria/.local/bin"})
    return json.loads(out_json.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="16,24,32,40,48,56")
    ap.add_argument("--out-dir", default="/home/rkathuria/manifolds/figures/sir_sweep")
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",")]

    rows = []
    for L in layers:
        try:
            res = run_one(L, out_dir)
        except Exception as e:
            print(f"layer {L} failed: {e}")
            continue
        row = {"layer": L,
               "supervised_nn": res["supervised"]["nn_accuracy"],
               "supervised_cyccos": res["supervised"]["cyclic_order_cos_corr"],
               "is_monday_nn": res["pos_is_monday"]["nn_accuracy"],
               "is_monday_cyccos": res["pos_is_monday"]["cyclic_order_cos_corr"],
               "is_monday_proc": res["pos_is_monday"]["procrustes"],
               "is_wednesday_nn": res["pos_is_wednesday"]["nn_accuracy"],
               "is_wednesday_cyccos": res["pos_is_wednesday"]["cyclic_order_cos_corr"],
               "is_wednesday_proc": res["pos_is_wednesday"]["procrustes"],
               "random_nn_mean": res["neg_random"]["nn_accuracy_mean"],
               "random_nn_std": res["neg_random"]["nn_accuracy_std"],
               "random_proc_mean": res["neg_random"]["procrustes_mean"],
               "random_cyccos_mean": res["neg_random"]["cyclic_order_cos_corr_mean"],
               "random_cyccos_std": res["neg_random"]["cyclic_order_cos_corr_std"],
               }
        if "neg_refusal_direction_dense" in res:
            row["refusal_nn"] = res["neg_refusal_direction_dense"]["nn_accuracy"]
            row["refusal_proc"] = res["neg_refusal_direction_dense"]["procrustes"]
        if "neg_testing_vs_conversation_opus46_v2" in res:
            row["tvc_nn"] = res["neg_testing_vs_conversation_opus46_v2"]["nn_accuracy"]
            row["tvc_proc"] = res["neg_testing_vs_conversation_opus46_v2"]["procrustes"]
        rows.append(row)

    Path(out_dir, "summary.json").write_text(json.dumps(rows, indent=2))
    print("\n" + "=" * 100)
    print(f"{'layer':>6} {'sup-NN':>8} {'monNN':>8} {'wedNN':>8} "
          f"{'rand-NN':>14} {'refNN':>8} {'tvcNN':>8} {'rand-cyc':>14}")
    for r in rows:
        rand = f"{r['random_nn_mean']:.3f}±{r['random_nn_std']:.3f}"
        rcyc = f"{r['random_cyccos_mean']:.3f}±{r['random_cyccos_std']:.3f}"
        print(f"{r['layer']:>6} {r['supervised_nn']:>8.3f} "
              f"{r['is_monday_nn']:>8.3f} {r['is_wednesday_nn']:>8.3f} "
              f"{rand:>14} "
              f"{r.get('refusal_nn', float('nan')):>8.3f} "
              f"{r.get('tvc_nn', float('nan')):>8.3f} "
              f"{rcyc:>14}")

    # Plot: NN-accuracy by layer for each probe condition
    layers_arr = np.array([r["layer"] for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(layers_arr, [r["supervised_nn"] for r in rows], "o-", label="supervised PCA (truth)", color="black")
    ax[0].plot(layers_arr, [r["is_monday_nn"] for r in rows], "o-", label="is_Monday probe", color="C2")
    ax[0].plot(layers_arr, [r["is_wednesday_nn"] for r in rows], "o-", label="is_Wednesday probe", color="C0")
    ax[0].errorbar(layers_arr, [r["random_nn_mean"] for r in rows],
                   yerr=[r["random_nn_std"] for r in rows], fmt="s--",
                   label="random probe (×20)", color="gray", alpha=0.7)
    if "refusal_nn" in rows[0]:
        ax[0].plot(layers_arr, [r.get("refusal_nn", np.nan) for r in rows], "x:",
                   label="refusal probe", color="C3")
    if "tvc_nn" in rows[0]:
        ax[0].plot(layers_arr, [r.get("tvc_nn", np.nan) for r in rows], "x:",
                   label="testing-vs-conv probe", color="C1")
    ax[0].axhline(1/7, ls="--", color="lightgray", label="chance (1/7)")
    ax[0].set_xlabel("layer"); ax[0].set_ylabel("nearest-centroid accuracy")
    ax[0].set_title("Cycle-recovery quality vs layer depth")
    ax[0].legend(loc="best", fontsize=8)
    ax[0].grid(alpha=0.3)

    ax[1].plot(layers_arr, [r["supervised_cyccos"] for r in rows], "o-", color="black", label="supervised")
    ax[1].plot(layers_arr, [r["is_monday_cyccos"] for r in rows], "o-", color="C2", label="is_Monday")
    ax[1].plot(layers_arr, [r["is_wednesday_cyccos"] for r in rows], "o-", color="C0", label="is_Wednesday")
    ax[1].errorbar(layers_arr, [r["random_cyccos_mean"] for r in rows],
                   yerr=[r["random_cyccos_std"] for r in rows], fmt="s--", color="gray",
                   alpha=0.7, label="random ×20")
    ax[1].set_xlabel("layer"); ax[1].set_ylabel("cyclic-cos-correlation (partner plane)")
    ax[1].set_title("Cyclic-order correlation vs layer (note: ambient cycle inflates negatives)")
    ax[1].legend(loc="best", fontsize=8)
    ax[1].grid(alpha=0.3)

    fig.suptitle("SIR slice+local-PCA recovery depth sweep   (v3 paper-faithful weekday data)")
    fig.tight_layout()
    fig.savefig(out_dir / "depth_sweep.png", dpi=140)
    plt.close(fig)
    print(f"\nwrote {out_dir}/depth_sweep.png and per-layer figures")


if __name__ == "__main__":
    main()

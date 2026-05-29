"""SAVE & DR on Llama-3.1-Tulu-3-8B-SFT (Llama-3.1 architecture).

Same paper-faithful prompts (temperature, age, year, days), Llama layers
12/16/19/22/26 (paper used L19). Compare to OLMo-3.1-32B results.
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
from dr_recovery import dr_recovery
from load_acts import load_labels, read_last_token_per_seq
from save_recovery import save_recovery


def chord_probe(X, values, q_low=0.2, q_high=0.8):
    arr = np.array(values, dtype=np.float64)
    lo = np.quantile(arr, q_low); hi = np.quantile(arr, q_high)
    w = X[arr >= hi].mean(axis=0) - X[arr <= lo].mean(axis=0)
    return w / (np.linalg.norm(w) + 1e-12)


def per_value_centroids(emb, values):
    uniq = sorted(set(values))
    return uniq, np.stack([emb[np.array(values) == v].mean(axis=0) for v in uniq])


def main():
    layers = [12, 16, 19, 22, 26]
    SHARD = "/data/artifacts/rohan/manifolds/paper_llama31_8b_v4/shard_dp00"
    LABELS = "/home/rkathuria/manifolds/data/labels_llama_paper.jsonl"

    labels = load_labels(Path(LABELS))
    weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

    rng = np.random.default_rng(0)
    rows = []
    for L in layers:
        X_full, pids = read_last_token_per_seq(Path(SHARD), L)

        for fam in ["temperature", "age", "year", "weekday"]:
            keep = [i for i, p in enumerate(pids) if labels[p].get("family") == fam]
            if not keep:
                continue
            X = X_full[keep]
            if fam == "weekday":
                values = [weekday_order.index(labels[pids[i]]["concept"]) for i in keep]
                fam_label = "days"
            else:
                values = [int(labels[pids[i]]["value"]) for i in keep]
                fam_label = fam

            sup_emb = PCA(n_components=3).fit_transform(X)
            _, sup_cents = per_value_centroids(sup_emb, values)

            w = chord_probe(X, values)
            sv = save_recovery(X, w, n_slices=10, d_M=2)
            dr = dr_recovery(X, w, n_slices=10, d_M=2)
            _, save_cents = per_value_centroids(sv.embedding, values)
            _, dr_cents = per_value_centroids(dr.embedding, values)

            _, _, save_p = procrustes(sup_cents, save_cents)
            _, _, dr_p = procrustes(sup_cents, dr_cents)

            # Random baseline (5 trials, SAVE only — to compute spectral gap)
            rand_top_eigs = []
            rand_save_procs = []
            for _ in range(5):
                wr = rng.standard_normal(X.shape[1]); wr /= np.linalg.norm(wr)
                sv_r = save_recovery(X, wr, n_slices=10, d_M=2)
                _, rc = per_value_centroids(sv_r.embedding, values)
                _, _, rp = procrustes(sup_cents, rc)
                rand_top_eigs.append(sv_r.eigvals[0])
                rand_save_procs.append(rp)

            rows.append({
                "model": "Llama-3.1-Tulu-3-8B-SFT", "layer": L, "family": fam_label,
                "n": int(X.shape[0]),
                "save_proc": float(save_p), "dr_proc": float(dr_p),
                "save_top3": [float(x) for x in sv.eigvals[:3]],
                "dr_top3": [float(x) for x in dr.eigvals[:3]],
                "rand_proc_mean": float(np.mean(rand_save_procs)),
                "spectral_gap": float(sv.eigvals[0] / np.mean(rand_top_eigs)),
                "c_balance": dr.c_balance,
            })
            print(f"[L{L:02d} {fam_label:>11}] SAVE={save_p:.4f}  DR={dr_p:.4f}  "
                  f"rand={np.mean(rand_save_procs):.4f}  gap={rows[-1]['spectral_gap']:.2f}×")

    out = Path("/home/rkathuria/manifolds/figures/llama_save_dr.json")
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")

    # Plot summary
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    families = sorted({r["family"] for r in rows})
    colors = {f: f"C{i}" for i, f in enumerate(families)}
    for fam in families:
        rs = [r for r in rows if r["family"] == fam]
        ls = [r["layer"] for r in rs]
        axes[0].plot(ls, [r["save_proc"] for r in rs], "o-", color=colors[fam],
                      label=f"{fam} SAVE")
        axes[0].plot(ls, [r["dr_proc"] for r in rs], "s--", color=colors[fam],
                      label=f"{fam} DR")
        axes[1].plot(ls, [r["spectral_gap"] for r in rs], "o-",
                      color=colors[fam], label=fam)
    axes[0].axvline(19, color="gray", linestyle=":", label="paper L19")
    axes[1].axvline(19, color="gray", linestyle=":")
    axes[1].axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    axes[0].set_xlabel("layer"); axes[0].set_ylabel("Procrustes")
    axes[0].set_title("Llama-3.1-Tulu-3-8B-SFT  SAVE vs DR (lower = better)")
    axes[0].legend(loc="best", fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("layer"); axes[1].set_ylabel("spectral gap (>1 = probe matters)")
    axes[1].set_title("Probe-faithfulness spectral gap")
    axes[1].legend(loc="best", fontsize=8); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("/home/rkathuria/manifolds/figures/llama_save_dr.png", dpi=140)
    plt.close(fig)
    print("wrote /home/rkathuria/manifolds/figures/llama_save_dr.png")


if __name__ == "__main__":
    main()

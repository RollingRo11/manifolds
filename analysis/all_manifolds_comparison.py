"""Single unified figure: per-manifold rows, two columns (supervised vs SAVE)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
    return uniq, np.stack([emb[np.array(values) == v].mean(axis=0) for v in uniq])


def main():
    layer = 16
    labels_v4 = load_labels(Path("/home/rkathuria/manifolds/data/labels_v4_paper.jsonl"))
    X4_full, pids4 = read_last_token_per_seq(
        Path("/data/artifacts/rohan/manifolds/paper_olmo31_32b_v4/shard_dp00"), layer)

    labels_v3 = load_labels(Path("/home/rkathuria/manifolds/data/labels_v3.jsonl"))
    X3_full, pids3 = read_last_token_per_seq(
        Path("/data/artifacts/rohan/manifolds/concepts_olmo31_32b_v3/shard_dp00"), layer)
    weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

    rows = []
    for fam, label_map, geometry in [
        ("temperature", lambda p: int(labels_v4[p]["value"]), "line"),
        ("age", lambda p: int(labels_v4[p]["value"]), "line"),
        ("year", lambda p: int(labels_v4[p]["value"]), "helix"),
    ]:
        keep = [i for i, p in enumerate(pids4) if labels_v4[p]["family"] == fam]
        X = X4_full[keep]
        values = [label_map(pids4[i]) for i in keep]
        rows.append((fam, geometry, X, values))

    keep = [i for i, p in enumerate(pids3) if labels_v3[p]["family"] == "weekday"]
    Xw = X3_full[keep]
    valuesw = [weekday_order.index(labels_v3[pids3[i]]["concept"]) for i in keep]
    rows.append(("days", "circle", Xw, valuesw))

    # --- Big figure: 4 rows × 2 cols ---
    fig = plt.figure(figsize=(14, 22))
    cmap = plt.get_cmap("viridis")
    for ri, (fam, geometry, X, values) in enumerate(rows):
        # supervised
        sup_emb = PCA(n_components=3).fit_transform(X)
        uniq_s, sup_cents = per_value_centroids(sup_emb, values)

        # SAVE
        w = chord_probe_extremes(X, values)
        sv = save_recovery(X, w, n_slices=10, d_M=2)
        uniq_v, save_cents = per_value_centroids(sv.embedding, values)

        norm = plt.Normalize(min(values), max(values))
        ax_l = fig.add_subplot(4, 2, 2 * ri + 1, projection="3d")
        ax_l.scatter(sup_emb[:, 0], sup_emb[:, 1], sup_emb[:, 2],
                      c=values, cmap=cmap, norm=norm, s=20, alpha=0.7)
        ax_l.plot(sup_cents[:, 0], sup_cents[:, 1], sup_cents[:, 2],
                   "k-", alpha=0.5, lw=1.0)
        ax_l.set_title(f"{fam.upper()} ({geometry}, n={X.shape[0]})\n"
                       "SUPERVISED PCA — what the paper gets", fontsize=12)
        ax_l.set_xlabel("PC1"); ax_l.set_ylabel("PC2"); ax_l.set_zlabel("PC3")

        ax_r = fig.add_subplot(4, 2, 2 * ri + 2, projection="3d")
        sc = ax_r.scatter(sv.embedding[:, 0], sv.embedding[:, 1], sv.embedding[:, 2],
                           c=values, cmap=cmap, norm=norm, s=20, alpha=0.7)
        ax_r.plot(save_cents[:, 0], save_cents[:, 1], save_cents[:, 2],
                   "k-", alpha=0.5, lw=1.0)
        ax_r.set_title(f"SAVE RECOVERED  (probe = high-extreme − low-extreme)\n"
                       f"top-eigvals={[round(x,1) for x in sv.eigvals[:3]]}",
                       fontsize=12)
        ax_r.set_xlabel("s = ŵ·h"); ax_r.set_ylabel("SAVE dir 1"); ax_r.set_zlabel("SAVE dir 2")

    fig.suptitle("Sanity test: supervised PCA (what we should get) vs SAVE-recovered (from a single probe)\n"
                 "OLMo-3.1-32B-Think  layer 16  paper-faithful prompts",
                 fontsize=14, y=0.995)
    fig.tight_layout()
    out = Path("/home/rkathuria/manifolds/figures/all_manifolds_supervised_vs_save.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

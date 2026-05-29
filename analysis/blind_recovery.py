"""Blind probe-driven manifold recovery + Procrustes evaluation.

Algorithm (deterministic — no labels used):

    s        = (X @ w) / ||w||                         # probe coord
    R        = X - s · ŵ                                # orthogonal residual
    Rm       = R restricted to {|s - median(s)| < ε}    # mid-slab
    V        = top-k PCs of Rm                          # partner axes
    embed    = (s, R @ V[0], R @ V[1], ...)             # fixed embedding

The supervised "ground truth" is the top-(1+k) PCs of the same activations.

Evaluation: Procrustes alignment (best rigid rotation+scale matching the two
embeddings) produces a residual error. Lower = better match.

We compare:
  - chord probe trained ON SAME data         (positive control)
  - chord probe trained on DIFFERENT data    (cross-set positive)
  - random unit vector                       (negative control: noise floor)
  - unrelated safety probe (refusal etc.)    (negative control: real)

A working benchmark: positive controls give residual << random baseline;
negative controls give residual ≈ random baseline.
"""
from __future__ import annotations

import argparse
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
from supervised import CALENDAR_ORDER


def restrict_to_family(X, pids, labels, family):
    keep = [i for i, p in enumerate(pids) if labels[p]["family"] == family]
    return X[keep], [pids[i] for i in keep]


def chord_probe(X, pids, labels, a, b):
    A = np.array([i for i, p in enumerate(pids) if labels[p]["concept"] == a])
    B = np.array([i for i, p in enumerate(pids) if labels[p]["concept"] == b])
    w = X[A].mean(axis=0) - X[B].mean(axis=0)
    return w / (np.linalg.norm(w) + 1e-12)


def blind_recover(X: np.ndarray, w: np.ndarray, n_partner: int = 2,
                  mid_quantile: float = 0.4) -> np.ndarray:
    """Deterministic blind recovery: returns [N, 1+n_partner] embedding.

    No labels. Algorithm is a fixed function of (X, w).
    """
    w = w.astype(np.float64); w = w / (np.linalg.norm(w) + 1e-12)
    s = X @ w
    R = X - s[:, None] * w[None, :]
    s_med = float(np.median(s))
    iqr = float(np.subtract(*np.quantile(s, [0.75, 0.25])))
    half = mid_quantile * iqr / 2 + 1e-9
    mask = np.abs(s - s_med) < half
    if mask.sum() < n_partner + 5:
        mask = np.ones(R.shape[0], dtype=bool)
    Rm = R[mask] - R[mask].mean(axis=0)
    _, _, Vt = np.linalg.svd(Rm, full_matrices=False)
    V = Vt[:n_partner]                      # [n_partner, d]
    coords = R @ V.T                        # [N, n_partner]
    return np.column_stack([s, coords])


def supervised_embedding(X: np.ndarray, n_dims: int = 3) -> np.ndarray:
    """Top-n_dims PCs of activations — paper's ground-truth manifold."""
    return PCA(n_components=n_dims).fit_transform(X)


def procrustes_residual(A: np.ndarray, B: np.ndarray) -> float:
    """scipy.spatial.procrustes returns (mtx1_std, mtx2_std, disparity).
    disparity = sum of squared differences after best rigid transform.
    Both inputs are first standardized (mean 0, unit Frobenius norm)."""
    if A.shape[1] != B.shape[1]:
        # Pad the smaller embedding with zeros so Procrustes can run.
        d = max(A.shape[1], B.shape[1])
        A = np.column_stack([A, np.zeros((A.shape[0], d - A.shape[1]))])
        B = np.column_stack([B, np.zeros((B.shape[0], d - B.shape[1]))])
    _, _, disp = procrustes(A, B)
    return float(disp)


def per_concept_centroids(emb: np.ndarray, cidx: np.ndarray, C: int) -> np.ndarray:
    return np.stack([emb[cidx == k].mean(axis=0) for k in range(C)])


def f_statistic(coord: np.ndarray, cidx: np.ndarray, C: int) -> float:
    """Between-class variance / within-class variance for a 1D coord."""
    grand = coord.mean()
    within = 0.0
    between = 0.0
    n_total = len(coord)
    for k in range(C):
        m = cidx == k
        nk = int(m.sum())
        if nk == 0: continue
        cm = coord[m].mean()
        between += nk * (cm - grand) ** 2
        within += float(((coord[m] - cm) ** 2).sum())
    df_b = C - 1
    df_w = max(n_total - C, 1)
    return float((between / df_b) / (within / df_w + 1e-12))


def nearest_centroid_accuracy(emb: np.ndarray, cidx: np.ndarray, C: int) -> float:
    """Classify each point to its nearest centroid (in the embedding); compare
    to the true concept label. Chance = 1/C for balanced classes."""
    centroids = np.stack([emb[cidx == k].mean(axis=0) for k in range(C)])
    # squared distances
    d2 = ((emb[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    pred = np.argmin(d2, axis=1)
    return float((pred == cidx).mean())


def evaluate_recovery(emb: np.ndarray, cidx: np.ndarray, C: int) -> dict:
    f_per_axis = [f_statistic(emb[:, k], cidx, C) for k in range(emb.shape[1])]
    return {
        "f_stat_per_axis": f_per_axis,
        "f_stat_min": float(min(f_per_axis)),
        "nearest_centroid_accuracy": nearest_centroid_accuracy(emb, cidx, C),
        "chance_accuracy": 1.0 / C,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2-shard",
                    default="/data/artifacts/rohan/manifolds/concepts_olmo31_32b_v2/shard_dp00")
    ap.add_argument("--v2-labels", default="/home/rkathuria/manifolds/data/labels_v2.jsonl")
    ap.add_argument("--v3-shard",
                    default="/data/artifacts/rohan/manifolds/concepts_olmo31_32b_v3/shard_dp00")
    ap.add_argument("--v3-labels", default="/home/rkathuria/manifolds/data/labels_v3.jsonl")
    ap.add_argument("--probes-dir", default="/data/artifacts/rohan/santi/probes")
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--n-partner", type=int, default=2,
                    help="how many partner PCs to extract (recovery dim = 1 + n_partner)")
    ap.add_argument("--out-png", default="/home/rkathuria/manifolds/figures/blind_recovery.png")
    ap.add_argument("--out-json",
                    default="/home/rkathuria/manifolds/figures/blind_recovery.json")
    args = ap.parse_args()

    labels_v2 = load_labels(Path(args.v2_labels))
    labels_v3 = load_labels(Path(args.v3_labels))
    order = CALENDAR_ORDER["weekday"]
    C = len(order)

    # Load both shards' weekday activations
    Xv2_all, pids_v2 = read_last_token_per_seq(Path(args.v2_shard), args.layer)
    Xv2, pids_v2 = restrict_to_family(Xv2_all, pids_v2, labels_v2, "weekday")
    Xv3_all, pids_v3 = read_last_token_per_seq(Path(args.v3_shard), args.layer)
    Xv3, pids_v3 = restrict_to_family(Xv3_all, pids_v3, labels_v3, "weekday")
    cidx_v3 = np.array([order.index(labels_v3[p]["concept"]) for p in pids_v3])
    print(f"v2 weekday n={Xv2.shape[0]}  v3 weekday n={Xv3.shape[0]}")

    # === Ground-truth supervised embedding (for v3) ============================
    sup_dim = 1 + args.n_partner
    sup_emb = supervised_embedding(Xv3, n_dims=sup_dim)
    sup_centroids = per_concept_centroids(sup_emb, cidx_v3, C)
    sup_eval = evaluate_recovery(sup_emb, cidx_v3, C)
    print(f"\nsupervised (paper-style) PCA top-{sup_dim} on v3 weekday data — ground truth")
    print(f"  f-stats per axis = {[f'{x:.1f}' for x in sup_eval['f_stat_per_axis']]}")
    print(f"  nearest-centroid accuracy = {sup_eval['nearest_centroid_accuracy']:.3f}  "
          f"(chance {sup_eval['chance_accuracy']:.3f})")

    # === Probes to compare =====================================================
    results = {}

    def add_result(key, name, w):
        rec = blind_recover(Xv3, w, n_partner=args.n_partner)
        rec_centroids = per_concept_centroids(rec, cidx_v3, C)
        ev = evaluate_recovery(rec, cidx_v3, C)
        results[key] = {
            "probe": name,
            "procrustes_disparity": procrustes_residual(rec_centroids, sup_centroids),
            "f_stat_per_axis": ev["f_stat_per_axis"],
            "f_stat_min": ev["f_stat_min"],
            "nearest_centroid_accuracy": ev["nearest_centroid_accuracy"],
            "rec_centroids": rec_centroids.tolist(),
        }

    # 1. POSITIVE: chord probe on v3 (same-set)
    w = chord_probe(Xv3, pids_v3, labels_v3, "Monday", "Friday")
    add_result("pos_chord_v3_to_v3", "chord(Mon,Fri) on v3", w)

    # 2. CROSS-SET POSITIVE: chord probe on v2, recovery on v3
    w = chord_probe(Xv2, pids_v2, labels_v2, "Monday", "Friday")
    add_result("pos_chord_v2_to_v3", "chord(Mon,Fri) on v2 → v3", w)

    # 3. NEGATIVE: random unit vector (noise floor) — collect all metrics
    rng = np.random.default_rng(0)
    rand_disps, rand_acc, rand_fmin = [], [], []
    for trial in range(20):
        w_rand = rng.standard_normal(Xv3.shape[1]).astype(np.float32)
        w_rand /= np.linalg.norm(w_rand)
        rec = blind_recover(Xv3, w_rand, n_partner=args.n_partner)
        rec_centroids = per_concept_centroids(rec, cidx_v3, C)
        ev = evaluate_recovery(rec, cidx_v3, C)
        rand_disps.append(procrustes_residual(rec_centroids, sup_centroids))
        rand_acc.append(ev["nearest_centroid_accuracy"])
        rand_fmin.append(ev["f_stat_min"])
    results["neg_random_baseline"] = {
        "probe": "random unit vector × 20 trials",
        "procrustes_disparity_mean": float(np.mean(rand_disps)),
        "procrustes_disparity_std": float(np.std(rand_disps)),
        "nearest_centroid_accuracy_mean": float(np.mean(rand_acc)),
        "nearest_centroid_accuracy_std": float(np.std(rand_acc)),
        "f_stat_min_mean": float(np.mean(rand_fmin)),
        "f_stat_min_std": float(np.std(rand_fmin)),
    }

    # 4-6. NEGATIVE controls: probes from elsewhere
    for fname, probe_key, label in [
        ("refusal_direction_dense.npz", f"L{args.layer}_refusal_direction",
         f"refusal_direction @ L{args.layer}"),
        ("testing_vs_conversation_opus46_v2.npz",
         f"L{args.layer}_testing_vs_conversation_opus46",
         f"testing_vs_conversation @ L{args.layer}"),
        ("fiction_vs_real_v2.npz", f"L{args.layer}_fiction_vs_real",
         f"fiction_vs_real @ L{args.layer}"),
    ]:
        npz_path = Path(args.probes_dir) / fname
        if not npz_path.exists():
            continue
        npz = np.load(npz_path, allow_pickle=False)
        if probe_key not in npz.files:
            continue
        w = npz[probe_key].astype(np.float32)
        add_result(f"neg_{fname.replace('.npz', '')}", label, w)

    # ===== summary table =======================================================
    print("\n" + "=" * 100)
    print(f"{'probe':<48} {'NN-acc':>8} {'F_min':>8} {'F_axes':>20} {'procrustes':>12}")
    print(f"{'  supervised top-3 PCA (ground truth)':<48} "
          f"{sup_eval['nearest_centroid_accuracy']:>8.3f} "
          f"{sup_eval['f_stat_min']:>8.2f} "
          f"{str([round(x,1) for x in sup_eval['f_stat_per_axis']]):>20}  (n/a)")
    print("=" * 100)
    for k, v in results.items():
        if "nearest_centroid_accuracy_mean" in v:
            acc = f"{v['nearest_centroid_accuracy_mean']:.3f}±{v['nearest_centroid_accuracy_std']:.3f}"
            fm = f"{v['f_stat_min_mean']:.2f}±{v['f_stat_min_std']:.2f}"
            disp = f"{v['procrustes_disparity_mean']:.3f}±{v['procrustes_disparity_std']:.3f}"
            print(f"{v['probe']:<48} {acc:>8} {fm:>8} {'(20 trials)':>20} {disp:>12}")
        else:
            faxes = str([round(x, 1) for x in v["f_stat_per_axis"]])
            print(f"{v['probe']:<48} {v['nearest_centroid_accuracy']:>8.3f} "
                  f"{v['f_stat_min']:>8.2f} {faxes:>20} {v['procrustes_disparity']:>12.4f}")
    rand_acc = results["neg_random_baseline"]["nearest_centroid_accuracy_mean"]
    rand_acc_sd = results["neg_random_baseline"]["nearest_centroid_accuracy_std"]
    print(f"\nNoise floor: random-probe NN-accuracy = {rand_acc:.3f} ± {rand_acc_sd:.3f}  "
          f"(chance = {1/C:.3f})")
    print(f"  positive probe should give NN-accuracy approaching supervised "
          f"({sup_eval['nearest_centroid_accuracy']:.3f}); negatives near random.")

    Path(args.out_json).write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out_json}")

    # ===== visualize ===========================================================
    cmap = plt.get_cmap("hsv")

    def plot_panel(ax, centroids, title, sup=False):
        cents2 = centroids[:, :2]
        for k in range(C):
            ax.scatter(cents2[k, 0], cents2[k, 1], s=240, color=cmap(k / C),
                       edgecolor="black", linewidth=1.5, zorder=10)
            ax.annotate(order[k], cents2[k] + np.array([0.02, 0.02]) * np.ptp(cents2),
                        fontsize=9)
        loop = np.vstack([cents2, cents2[:1]])
        ax.plot(loop[:, 0], loop[:, 1], "k--", lw=1.0, alpha=0.5)
        ax.set_title(title, fontsize=11)
        ax.set_aspect("equal", "datalim")

    panels = [(f"supervised PCA (truth)\nNN-acc={sup_eval['nearest_centroid_accuracy']:.3f}",
               sup_centroids)]
    for k, v in results.items():
        if "rec_centroids" not in v:
            continue
        label = (f"{v['probe']}\n"
                 f"NN-acc={v['nearest_centroid_accuracy']:.3f}  F_min={v['f_stat_min']:.2f}")
        panels.append((label, np.array(v["rec_centroids"])))

    n = len(panels)
    fig = plt.figure(figsize=(5 * n, 5))
    for i, (title, cent) in enumerate(panels):
        ax = fig.add_subplot(1, n, i + 1)
        plot_panel(ax, cent, title)
    fig.suptitle(f"Blind probe recovery vs supervised PCA — layer {args.layer}\n"
                 f"random-probe noise floor: NN-acc = {rand_acc:.3f} ± {rand_acc_sd:.3f}   "
                 f"(chance = {1/C:.3f}, supervised = {sup_eval['nearest_centroid_accuracy']:.3f})")
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=140)
    plt.close(fig)
    print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()

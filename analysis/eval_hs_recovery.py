"""Compare HS principal-curve recovery (v2) to plain SIR (v1) and supervised PCA.

Loss function: Procrustes disparity between the recovery's per-concept
centroids and supervised PCA's per-concept centroids. Lower = better.
The "negative control" probes (refusal etc.) should NOT improve as the
algorithm gets better — the algorithm should only get better at the
intended job (recovering the structure for an informative probe).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import procrustes
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent))
from load_acts import load_labels, read_last_token_per_seq
from sir_recovery import binary_probe_diff_of_means, slice_and_local_pca
from sir_recovery_v2 import hs_principal_curve
from sir_recovery_v3 import global_residual_recover
from sir_recovery_v4 import sir_proper
from sir_recovery_v5 import smoothed_per_bin_sir
from sir_recovery_v6 import reduced_per_bin_sir
from sir_recovery_v7 import kernel_local_pca_sir
from supervised import CALENDAR_ORDER


def per_concept_centroids(emb, cidx, C):
    return np.stack([emb[cidx == k].mean(axis=0) for k in range(C)])


def nn_acc(emb, cidx, C):
    cents = per_concept_centroids(emb, cidx, C)
    d2 = ((emb[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2)
    return float((np.argmin(d2, axis=1) == cidx).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir",
                    default="/data/artifacts/rohan/manifolds/concepts_olmo31_32b_v3/shard_dp00")
    ap.add_argument("--labels", default="/home/rkathuria/manifolds/data/labels_v3.jsonl")
    ap.add_argument("--probes-dir", default="/data/artifacts/rohan/santi/probes")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--probe-day", default="Wednesday")
    ap.add_argument("--n-bins", type=int, default=20)
    ap.add_argument("--n-partner", type=int, default=2)
    ap.add_argument("--n-iter", type=int, default=5)
    ap.add_argument("--smooth-sigma", type=float, default=1.0)
    ap.add_argument("--out-json", default="/home/rkathuria/manifolds/figures/hs_eval.json")
    args = ap.parse_args()

    labels = load_labels(Path(args.labels))
    X_full, pids = read_last_token_per_seq(Path(args.shard_dir), args.layer)
    keep = [i for i, p in enumerate(pids) if labels[p]["family"] == "weekday"]
    X = X_full[keep]
    pids = [pids[i] for i in keep]
    order = CALENDAR_ORDER["weekday"]
    C = len(order)
    cidx = np.array([order.index(labels[p]["concept"]) for p in pids])

    sup_dim = 1 + args.n_partner
    sup_emb = PCA(n_components=sup_dim).fit_transform(X)
    sup_cents = per_concept_centroids(sup_emb, cidx, C)
    sup_nn = nn_acc(sup_emb, cidx, C)
    print(f"\nlayer {args.layer}  n={X.shape[0]}  supervised top-{sup_dim} PCA NN-acc = {sup_nn:.3f}")

    rng = np.random.default_rng(0)

    def evaluate(name, w, sample_n_random=0):
        # v1: plain SIR (per-bin local PCA)
        v1 = slice_and_local_pca(X, w, n_bins=args.n_bins, n_partner=args.n_partner)
        v1_cents = per_concept_centroids(v1.embedding, cidx, C)
        _, _, v1_disp = procrustes(sup_cents, v1_cents)
        v1_nn = nn_acc(v1.embedding, cidx, C)

        # v2: HS principal curve (iterative)
        v2 = hs_principal_curve(X, w, n_bins=args.n_bins, n_partner=args.n_partner,
                                 n_iter=args.n_iter, smooth_sigma=args.smooth_sigma)
        v2_cents = per_concept_centroids(v2.embedding, cidx, C)
        _, _, v2_disp = procrustes(sup_cents, v2_cents)
        v2_nn = nn_acc(v2.embedding, cidx, C)

        # v3: global residual-PCA partner axes
        v3_emb = global_residual_recover(X, w, n_partner=args.n_partner)
        v3_cents = per_concept_centroids(v3_emb, cidx, C)
        _, _, v3_disp = procrustes(sup_cents, v3_cents)
        v3_nn = nn_acc(v3_emb, cidx, C)

        # v4: formal SIR (between-bin scatter)
        v4_emb = sir_proper(X, w, n_bins=args.n_bins, n_partner=args.n_partner)
        v4_cents = per_concept_centroids(v4_emb, cidx, C)
        _, _, v4_disp = procrustes(sup_cents, v4_cents)
        v4_nn = nn_acc(v4_emb, cidx, C)

        # v5: per-bin SIR + Gaussian-smoothed tangents
        v5_emb = smoothed_per_bin_sir(X, w, n_bins=args.n_bins,
                                       n_partner=args.n_partner, smooth_sigma=2.0)
        v5_cents = per_concept_centroids(v5_emb, cidx, C)
        _, _, v5_disp = procrustes(sup_cents, v5_cents)
        v5_nn = nn_acc(v5_emb, cidx, C)

        # v6: v1 in pre-PCA-reduced subspace
        v6_emb = reduced_per_bin_sir(X, w, n_bins=args.n_bins,
                                      n_partner=args.n_partner, n_pca_pre=64)
        v6_cents = per_concept_centroids(v6_emb, cidx, C)
        _, _, v6_disp = procrustes(sup_cents, v6_cents)
        v6_nn = nn_acc(v6_emb, cidx, C)

        # v7: kernel-weighted local PCA in PCA-64 subspace
        v7_emb = kernel_local_pca_sir(X, w, n_eval=args.n_bins,
                                       n_partner=args.n_partner, n_pca_pre=64)
        v7_cents = per_concept_centroids(v7_emb, cidx, C)
        _, _, v7_disp = procrustes(sup_cents, v7_cents)
        v7_nn = nn_acc(v7_emb, cidx, C)

        print(f"\n[{name}]")
        print(f"  v1 per-bin SIR           : NN={v1_nn:.3f}  procrustes={v1_disp:.4f}")
        print(f"  v2 HS iter principal-curve: NN={v2_nn:.3f}  procrustes={v2_disp:.4f}")
        print(f"  v3 global residual-PCA   : NN={v3_nn:.3f}  procrustes={v3_disp:.4f}")
        print(f"  v4 SIR formal between-bin: NN={v4_nn:.3f}  procrustes={v4_disp:.4f}")
        print(f"  v5 smoothed per-bin SIR  : NN={v5_nn:.3f}  procrustes={v5_disp:.4f}")
        print(f"  v6 v1 in PCA-64 subspace : NN={v6_nn:.3f}  procrustes={v6_disp:.4f}")
        print(f"  v7 v6 + kernel weights   : NN={v7_nn:.3f}  procrustes={v7_disp:.4f}")
        return {"name": name,
                "v1_nn": v1_nn, "v1_procrustes": float(v1_disp),
                "v2_nn": v2_nn, "v2_procrustes": float(v2_disp),
                "v3_nn": v3_nn, "v3_procrustes": float(v3_disp),
                "v4_nn": v4_nn, "v4_procrustes": float(v4_disp),
                "v5_nn": v5_nn, "v5_procrustes": float(v5_disp),
                "v6_nn": v6_nn, "v6_procrustes": float(v6_disp),
                "v7_nn": v7_nn, "v7_procrustes": float(v7_disp)}

    results = {"supervised_nn": sup_nn, "probes": []}

    # Positive: is_<probe-day>
    pos_mask = np.array([labels[p]["concept"] == args.probe_day for p in pids])
    w = binary_probe_diff_of_means(X, pos_mask)
    results["probes"].append(evaluate(f"is_{args.probe_day}", w))

    # Random × 10
    rand_v1, rand_v2, rand_v3, rand_v4, rand_v5, rand_v6, rand_v7 = [], [], [], [], [], [], []
    for trial in range(10):
        wr = rng.standard_normal(X.shape[1]).astype(np.float32)
        wr /= np.linalg.norm(wr)
        v1 = slice_and_local_pca(X, wr, n_bins=args.n_bins, n_partner=args.n_partner)
        v1_cents = per_concept_centroids(v1.embedding, cidx, C)
        _, _, v1_disp = procrustes(sup_cents, v1_cents)
        v2 = hs_principal_curve(X, wr, n_bins=args.n_bins, n_partner=args.n_partner,
                                 n_iter=args.n_iter, smooth_sigma=args.smooth_sigma)
        v2_cents = per_concept_centroids(v2.embedding, cidx, C)
        _, _, v2_disp = procrustes(sup_cents, v2_cents)
        v3_emb = global_residual_recover(X, wr, n_partner=args.n_partner)
        v3_cents = per_concept_centroids(v3_emb, cidx, C)
        _, _, v3_disp = procrustes(sup_cents, v3_cents)
        v4_emb = sir_proper(X, wr, n_bins=args.n_bins, n_partner=args.n_partner)
        v4_cents = per_concept_centroids(v4_emb, cidx, C)
        _, _, v4_disp = procrustes(sup_cents, v4_cents)
        v5_emb = smoothed_per_bin_sir(X, wr, n_bins=args.n_bins,
                                       n_partner=args.n_partner, smooth_sigma=2.0)
        v5_cents = per_concept_centroids(v5_emb, cidx, C)
        _, _, v5_disp = procrustes(sup_cents, v5_cents)
        v6_emb = reduced_per_bin_sir(X, wr, n_bins=args.n_bins,
                                      n_partner=args.n_partner, n_pca_pre=64)
        v6_cents = per_concept_centroids(v6_emb, cidx, C)
        _, _, v6_disp = procrustes(sup_cents, v6_cents)
        v7_emb = kernel_local_pca_sir(X, wr, n_eval=args.n_bins,
                                       n_partner=args.n_partner, n_pca_pre=64)
        v7_cents = per_concept_centroids(v7_emb, cidx, C)
        _, _, v7_disp = procrustes(sup_cents, v7_cents)
        rand_v1.append((v1_disp, nn_acc(v1.embedding, cidx, C)))
        rand_v2.append((v2_disp, nn_acc(v2.embedding, cidx, C)))
        rand_v3.append((v3_disp, nn_acc(v3_emb, cidx, C)))
        rand_v4.append((v4_disp, nn_acc(v4_emb, cidx, C)))
        rand_v5.append((v5_disp, nn_acc(v5_emb, cidx, C)))
        rand_v6.append((v6_disp, nn_acc(v6_emb, cidx, C)))
        rand_v7.append((v7_disp, nn_acc(v7_emb, cidx, C)))
    rand_v1 = np.array(rand_v1); rand_v2 = np.array(rand_v2)
    rand_v3 = np.array(rand_v3); rand_v4 = np.array(rand_v4)
    rand_v5 = np.array(rand_v5); rand_v6 = np.array(rand_v6); rand_v7 = np.array(rand_v7)
    print(f"\n[random × 10]")
    print(f"  v1 per-bin SIR            : NN={rand_v1[:,1].mean():.3f}±{rand_v1[:,1].std():.3f}  "
          f"procrustes={rand_v1[:,0].mean():.4f}±{rand_v1[:,0].std():.4f}")
    print(f"  v2 HS iter principal-curve: NN={rand_v2[:,1].mean():.3f}±{rand_v2[:,1].std():.3f}  "
          f"procrustes={rand_v2[:,0].mean():.4f}±{rand_v2[:,0].std():.4f}")
    print(f"  v3 global residual-PCA    : NN={rand_v3[:,1].mean():.3f}±{rand_v3[:,1].std():.3f}  "
          f"procrustes={rand_v3[:,0].mean():.4f}±{rand_v3[:,0].std():.4f}")
    print(f"  v4 SIR formal between-bin : NN={rand_v4[:,1].mean():.3f}±{rand_v4[:,1].std():.3f}  "
          f"procrustes={rand_v4[:,0].mean():.4f}±{rand_v4[:,0].std():.4f}")
    print(f"  v5 smoothed per-bin SIR   : NN={rand_v5[:,1].mean():.3f}±{rand_v5[:,1].std():.3f}  "
          f"procrustes={rand_v5[:,0].mean():.4f}±{rand_v5[:,0].std():.4f}")
    print(f"  v6 v1 in PCA-64 subspace  : NN={rand_v6[:,1].mean():.3f}±{rand_v6[:,1].std():.3f}  "
          f"procrustes={rand_v6[:,0].mean():.4f}±{rand_v6[:,0].std():.4f}")
    print(f"  v7 v6 + kernel weights    : NN={rand_v7[:,1].mean():.3f}±{rand_v7[:,1].std():.3f}  "
          f"procrustes={rand_v7[:,0].mean():.4f}±{rand_v7[:,0].std():.4f}")
    results["random"] = {
        "v1_procrustes_mean": float(rand_v1[:,0].mean()),
        "v2_procrustes_mean": float(rand_v2[:,0].mean()),
        "v3_procrustes_mean": float(rand_v3[:,0].mean()),
        "v4_procrustes_mean": float(rand_v4[:,0].mean()),
        "v4_procrustes_std": float(rand_v4[:,0].std()),
        "v1_nn_mean": float(rand_v1[:,1].mean()),
        "v2_nn_mean": float(rand_v2[:,1].mean()),
        "v3_nn_mean": float(rand_v3[:,1].mean()),
        "v4_nn_mean": float(rand_v4[:,1].mean()),
    }

    # Unrelated: refusal, testing-vs-conv
    for fname, key, label in [
        ("refusal_direction_dense.npz", f"L{args.layer}_refusal_direction",
         f"refusal probe @ L{args.layer}"),
        ("testing_vs_conversation_opus46_v2.npz",
         f"L{args.layer}_testing_vs_conversation_opus46",
         f"testing_vs_conversation @ L{args.layer}"),
    ]:
        p = Path(args.probes_dir) / fname
        if not p.exists(): continue
        npz = np.load(p, allow_pickle=False)
        if key not in npz.files: continue
        results["probes"].append(evaluate(label, npz[key].astype(np.float32)))

    Path(args.out_json).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()

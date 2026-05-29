"""Evaluate SAVE on the v3 weekday benchmark — compare to v1, v6, v8 + sup PCA.

Also check the SAVE eigenvalue spectrum as a probe-faithfulness signal:
informative probes should have a clean spectral gap; random probes should
have eigenvalues collapsed at the noise floor.
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
from sir_recovery import binary_probe_diff_of_means, slice_and_local_pca
from sir_recovery_v3 import global_residual_recover
from sir_recovery_v6 import reduced_per_bin_sir
from sir_recovery_v8 import sir_subspace_per_bin
from supervised import CALENDAR_ORDER


def per_concept_centroids(emb, cidx, C):
    return np.stack([emb[cidx == k].mean(axis=0) for k in range(C)])


def nn_acc(emb, cidx, C):
    cents = per_concept_centroids(emb, cidx, C)
    d2 = ((emb[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2)
    return float((np.argmin(d2, axis=1) == cidx).mean())


def main():
    labels = load_labels(Path("/home/rkathuria/manifolds/data/labels_v3.jsonl"))
    X_full, pids = read_last_token_per_seq(
        Path("/data/artifacts/rohan/manifolds/concepts_olmo31_32b_v3/shard_dp00"), 16)
    keep = [i for i, p in enumerate(pids) if labels[p]["family"] == "weekday"]
    X = X_full[keep]
    pids = [pids[i] for i in keep]
    order = CALENDAR_ORDER["weekday"]; C = len(order)
    cidx = np.array([order.index(labels[p]["concept"]) for p in pids])
    sup_emb = PCA(n_components=3).fit_transform(X)
    sup_cents = per_concept_centroids(sup_emb, cidx, C)
    print(f"layer 16  n={X.shape[0]}  supervised top-3 NN={nn_acc(sup_emb, cidx, C):.3f}")

    # --- Positive probe: is_Wednesday
    pos = np.array([labels[p]["concept"] == "Wednesday" for p in pids])
    w_pos = binary_probe_diff_of_means(X, pos)

    # Run all algorithms
    def evaluate(w, label):
        results = {}
        # v1
        emb = slice_and_local_pca(X, w, n_bins=20, n_partner=2).embedding
        cents = per_concept_centroids(emb, cidx, C)
        _, _, disp = procrustes(sup_cents, cents)
        results["v1"] = {"nn": nn_acc(emb, cidx, C), "proc": float(disp)}
        # v6
        emb = reduced_per_bin_sir(X, w, n_bins=20, n_partner=2, n_pca_pre=64)
        cents = per_concept_centroids(emb, cidx, C)
        _, _, disp = procrustes(sup_cents, cents)
        results["v6"] = {"nn": nn_acc(emb, cidx, C), "proc": float(disp)}
        # v8
        emb = sir_subspace_per_bin(X, w, n_bins=20, n_partner=2, k_subspace=16)
        cents = per_concept_centroids(emb, cidx, C)
        _, _, disp = procrustes(sup_cents, cents)
        results["v8"] = {"nn": nn_acc(emb, cidx, C), "proc": float(disp)}
        # SAVE
        sv = save_recovery(X, w, n_slices=10, d_M=2)
        cents = per_concept_centroids(sv.embedding, cidx, C)
        _, _, disp = procrustes(sup_cents, cents)
        results["SAVE"] = {"nn": nn_acc(sv.embedding, cidx, C), "proc": float(disp),
                           "top5_eigs": sv.eigvals[:5].tolist()}
        # v3 (for ref)
        emb = global_residual_recover(X, w, n_partner=2)
        cents = per_concept_centroids(emb, cidx, C)
        _, _, disp = procrustes(sup_cents, cents)
        results["v3"] = {"nn": nn_acc(emb, cidx, C), "proc": float(disp)}
        return results, sv

    print("\n[is_Wednesday probe]")
    pos_res, sv_pos = evaluate(w_pos, "is_Wednesday")
    for k, v in pos_res.items():
        extra = f"  top-eigs={[round(x,4) for x in v['top5_eigs']]}" if "top5_eigs" in v else ""
        print(f"  {k:>5}: NN={v['nn']:.3f}  procrustes={v['proc']:.4f}{extra}")

    # --- Random × 10
    print("\n[random × 10]")
    rng = np.random.default_rng(0)
    rand_res = {k: {"nn": [], "proc": [], "top5_eigs": []}
                for k in ("v1", "v6", "v8", "SAVE", "v3")}
    for trial in range(10):
        wr = rng.standard_normal(X.shape[1]); wr /= np.linalg.norm(wr)
        rr, sv_r = evaluate(wr, "random")
        for k, v in rr.items():
            rand_res[k]["nn"].append(v["nn"])
            rand_res[k]["proc"].append(v["proc"])
            if "top5_eigs" in v:
                rand_res[k]["top5_eigs"].append(v["top5_eigs"])
    for k, v in rand_res.items():
        nns = np.array(v["nn"]); procs = np.array(v["proc"])
        print(f"  {k:>5}: NN={nns.mean():.3f}±{nns.std():.3f}  "
              f"procrustes={procs.mean():.4f}±{procs.std():.4f}")

    # --- Eigenvalue spectrum comparison: SAVE
    print("\n=== SAVE EIGENVALUE SPECTRUM (probe-faithfulness diagnostic) ===")
    pos_eigs = np.array(pos_res["SAVE"]["top5_eigs"])
    rand_eigs_mean = np.mean([e for e in rand_res["SAVE"]["top5_eigs"]], axis=0)
    rand_eigs_std = np.std([e for e in rand_res["SAVE"]["top5_eigs"]], axis=0)
    print(f"  is_Wednesday top-5 eigvals: {pos_eigs}")
    print(f"  random       top-5 eigvals: mean={rand_eigs_mean}")
    print(f"  random       top-5 eigvals: std ={rand_eigs_std}")
    pos_gap = pos_eigs[0] / max(pos_eigs[1], 1e-9)
    rand_gap = rand_eigs_mean[0] / max(rand_eigs_mean[1], 1e-9)
    print(f"  spectral gap (top1/top2):  is_Wed={pos_gap:.2f}  random={rand_gap:.2f}")

    # Save full spectra for plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    n_show = 10
    sv_pos = save_recovery(X, w_pos, n_slices=10, d_M=2)
    axes[0].bar(range(n_show), sv_pos.eigvals[:n_show])
    axes[0].set_title("SAVE eigenvalues — is_Wednesday probe")
    axes[0].set_xlabel("eigenvalue index"); axes[0].set_ylabel("magnitude")
    rand_eigs_full = []
    rng = np.random.default_rng(0)
    for _ in range(10):
        wr = rng.standard_normal(X.shape[1]); wr /= np.linalg.norm(wr)
        sv_r = save_recovery(X, wr, n_slices=10, d_M=2)
        rand_eigs_full.append(sv_r.eigvals[:n_show])
    rand_eigs_full = np.array(rand_eigs_full)
    axes[1].bar(range(n_show), rand_eigs_full.mean(axis=0),
                yerr=rand_eigs_full.std(axis=0), capsize=4)
    axes[1].set_title("SAVE eigenvalues — random probes (mean ± std × 10)")
    axes[1].set_xlabel("eigenvalue index")
    fig.tight_layout()
    fig.savefig("/home/rkathuria/manifolds/figures/save_spectrum.png", dpi=130)
    plt.close(fig)
    print("\nwrote /home/rkathuria/manifolds/figures/save_spectrum.png")


if __name__ == "__main__":
    main()

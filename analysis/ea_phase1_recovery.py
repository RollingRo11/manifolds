"""Phase 1: SAVE recovery of eval-awareness manifold on OLMo-32B.

Loads:
  - eval_awareness shard activations at chosen layer
  - existing eval-awareness probe (testing_vs_conversation_opus46)
  - frame_score labels (held out from recovery; only for evaluation)

Runs SAVE recovery, visualizes:
  - SAVE eigenvalue spectrum (probe-faithfulness diagnostic)
  - 3D embedding colored by frame_score
  - per-frame-score centroids on the recovered manifold
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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial import procrustes
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent))
from load_acts import load_labels, read_last_token_per_seq
from save_recovery import save_recovery


SHARD = Path("/data/artifacts/rohan/manifolds/eval_awareness_olmo31_32b_v1/shard_dp00")
LABELS = Path("/home/rkathuria/manifolds/data/eval_awareness_labels.jsonl")
PROBES_DIR = Path("/data/artifacts/rohan/santi/probes")


def load_probe(name: str, layer: int) -> np.ndarray | None:
    """Load a probe vector from any of the available .npz files."""
    candidates = {
        "testing_vs_conv": ("testing_vs_conversation_opus46_v2.npz",
                              f"L{layer}_testing_vs_conversation_opus46"),
        "hua_et_al": ("hua_et_al_v2.npz", f"L{layer}_hua_et_al"),
        "hua_et_al_ha": ("hua_et_al_ha_v2.npz", f"L{layer}_hua_et_al_ha"),
        "refusal": ("refusal_direction_dense.npz", f"L{layer}_refusal_direction"),
    }
    if name not in candidates:
        return None
    fname, key = candidates[name]
    p = PROBES_DIR / fname
    if not p.exists():
        return None
    npz = np.load(p, allow_pickle=False)
    if key not in npz.files:
        return None
    w = npz[key].astype(np.float32)
    return w / (np.linalg.norm(w) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=32)
    ap.add_argument("--probe", default="testing_vs_conv",
                    choices=["testing_vs_conv", "hua_et_al", "hua_et_al_ha",
                              "refusal", "diff_of_means"])
    ap.add_argument("--out-dir", default="/home/rkathuria/manifolds/figures/ea_phase1")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    labels = load_labels(LABELS)

    # Load activations
    X, pids = read_last_token_per_seq(SHARD, args.layer)
    fs = np.array([labels[p]["frame_score"] for p in pids])
    print(f"loaded {X.shape[0]} EA activations at L{args.layer}, d={X.shape[1]}")
    print(f"frame_score distribution: {np.bincount(fs)}")

    # Get probe
    if args.probe == "diff_of_means":
        # Train an in-house chord probe: high-frame-score minus low-frame-score
        high = X[fs >= 5].mean(0)
        low = X[fs <= 1].mean(0)
        w = high - low
        w /= np.linalg.norm(w)
        print(f"diff-of-means chord probe trained from frame_scores [≤1] vs [≥5]")
    else:
        w = load_probe(args.probe, args.layer)
        if w is None:
            raise SystemExit(f"probe {args.probe} not available at L{args.layer}")
        print(f"loaded probe '{args.probe}' at L{args.layer}")

    # ------ SAVE recovery ------
    sv = save_recovery(X, w, n_slices=10, d_M=2)
    print(f"\nSAVE eigenvalues (top 10): {[round(float(x),3) for x in sv.eigvals[:10]]}")
    print(f"  spectral gap top1/top2: {sv.eigvals[0] / max(sv.eigvals[1], 1e-9):.2f}")

    # Random probe baseline for spectrum comparison
    rng = np.random.default_rng(0)
    rand_top_eigs = []
    for _ in range(10):
        wr = rng.standard_normal(X.shape[1]); wr /= np.linalg.norm(wr)
        sv_r = save_recovery(X, wr, n_slices=10, d_M=2)
        rand_top_eigs.append(sv_r.eigvals[0])
    print(f"  random probe top eigval: {np.mean(rand_top_eigs):.3f} ± {np.std(rand_top_eigs):.3f}")
    print(f"  ratio (this probe / random): {sv.eigvals[0] / np.mean(rand_top_eigs):.2f}×")

    # ------ Per-frame-score centroids ------
    save_cents = np.stack([sv.embedding[fs == k].mean(0) for k in range(7) if (fs == k).sum() > 0])
    print(f"\nper-frame-score centroids in SAVE 3D embedding:")
    for k in range(7):
        m = fs == k
        if m.sum() == 0: continue
        c = sv.embedding[m].mean(0)
        print(f"  FS={k} (n={m.sum()}): s={c[0]:+.3f}  v1={c[1]:+.3f}  v2={c[2]:+.3f}")

    # ------ Supervised PCA on EA acts (for reference) ------
    sup_emb = PCA(n_components=3).fit_transform(X)
    sup_cents = np.stack([sup_emb[fs == k].mean(0) for k in range(7) if (fs == k).sum() > 0])
    _, _, proc = procrustes(sup_cents, save_cents)
    print(f"\nSAVE vs supervised-PCA Procrustes on FS centroids: {proc:.4f}")

    # ------ Diagnose dimensionality ------
    # Is the manifold 1D linear, curved 1D, or multi-D?
    # Compare to a perfect 1D linear baseline (probe coord alone)
    s = sv.s
    r_s_fs = float(np.corrcoef(s, fs)[0, 1])
    print(f"\nProbe-coord vs frame_score correlation: {r_s_fs:+.3f}")

    # Variance of partner axes after controlling for s
    from scipy.stats import pearsonr
    v1_resid = sv.embedding[:, 1] - r_s_fs * (fs - fs.mean())  # rough decomposition
    v2_resid = sv.embedding[:, 2]
    var_v1_explained_by_fs = float(np.corrcoef(sv.embedding[:, 1], fs)[0, 1])
    var_v2_explained_by_fs = float(np.corrcoef(sv.embedding[:, 2], fs)[0, 1])
    print(f"v1 vs frame_score correlation: {var_v1_explained_by_fs:+.3f}")
    print(f"v2 vs frame_score correlation: {var_v2_explained_by_fs:+.3f}")

    # ------ Visualizations ------
    # Eigenvalue spectrum plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    n_show = 10
    axes[0].bar(range(n_show), sv.eigvals[:n_show], color="C0", label=f"probe={args.probe}")
    axes[0].axhline(np.mean(rand_top_eigs), color="gray", linestyle="--",
                     label=f"random top-eig (mean of 10)")
    axes[0].set_title(f"SAVE eigenvalue spectrum  L{args.layer}, probe={args.probe}")
    axes[0].set_xlabel("eigenvalue index"); axes[0].set_ylabel("magnitude")
    axes[0].legend()

    axes[1].plot(range(7), [(fs == k).sum() for k in range(7)], "o-")
    axes[1].set_xlabel("frame_score"); axes[1].set_ylabel("# prompts")
    axes[1].set_title("EA shard distribution")
    fig.tight_layout()
    fig.savefig(out_dir / f"ea_phase1_diagnostics_L{args.layer}_{args.probe}.png", dpi=130)
    plt.close(fig)

    # Interactive 3D plot
    cmap = plt.get_cmap("plasma")
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}] * 2],
                         subplot_titles=(
                             f"<b>Supervised PCA</b> (paper-style, all activations)",
                             f"<b>SAVE recovered</b> from probe '{args.probe}' alone",))
    # supervised
    fig.add_trace(go.Scatter3d(x=sup_emb[:, 0], y=sup_emb[:, 1], z=sup_emb[:, 2],
        mode="markers", marker=dict(size=3, color=fs, colorscale="Plasma",
                                      colorbar=dict(title="frame_score", x=0.45),
                                      opacity=0.55, cmin=0, cmax=6),
        text=[f"FS={f}" for f in fs], hoverinfo="text", showlegend=False), row=1, col=1)
    sup_loop = np.vstack([sup_cents, sup_cents[:1]])
    fig.add_trace(go.Scatter3d(x=sup_loop[:, 0], y=sup_loop[:, 1], z=sup_loop[:, 2],
        mode="lines+markers", line=dict(color="black", width=4),
        marker=dict(size=8, color=list(range(7)), colorscale="Plasma", cmin=0, cmax=6,
                     line=dict(color="black", width=1)),
        showlegend=False), row=1, col=1)
    # recovered
    fig.add_trace(go.Scatter3d(x=sv.embedding[:, 0], y=sv.embedding[:, 1], z=sv.embedding[:, 2],
        mode="markers", marker=dict(size=3, color=fs, colorscale="Plasma",
                                      showscale=False, opacity=0.55, cmin=0, cmax=6),
        text=[f"FS={f}" for f in fs], hoverinfo="text", showlegend=False), row=1, col=2)
    save_loop = np.vstack([save_cents, save_cents[:1]])
    fig.add_trace(go.Scatter3d(x=save_loop[:, 0], y=save_loop[:, 1], z=save_loop[:, 2],
        mode="lines+markers", line=dict(color="black", width=4),
        marker=dict(size=8, color=list(range(7)), colorscale="Plasma", cmin=0, cmax=6,
                     line=dict(color="black", width=1)),
        showlegend=False), row=1, col=2)
    fig.update_layout(
        title=f"Eval-awareness manifold recovery — OLMo-3.1-32B-Think L{args.layer}, probe={args.probe}<br>"
              f"<sub>SAVE top-eigval ratio vs random: {sv.eigvals[0]/np.mean(rand_top_eigs):.2f}× | "
              f"Procrustes vs supervised PCA: {proc:.3f}</sub>",
        height=700, width=1500,
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
        scene2=dict(xaxis_title="s=ŵ·h", yaxis_title="SAVE 1", zaxis_title="SAVE 2"),
    )
    out_html = out_dir / f"ea_3d_L{args.layer}_{args.probe}.html"
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    print(f"\nwrote {out_html}")

    # Save data for steering phase
    np.savez(out_dir / f"ea_save_L{args.layer}_{args.probe}.npz",
              X=X, fs=fs, pids=np.array(pids),
              save_embedding=sv.embedding, save_eigvals=sv.eigvals,
              save_dirs=sv.save_dirs, save_V=sv.V, w=w,
              save_cents=save_cents, sup_cents=sup_cents,
              sup_pca_components=PCA(n_components=3).fit(X).components_)
    print(f"wrote ea_save_L{args.layer}_{args.probe}.npz")


if __name__ == "__main__":
    main()

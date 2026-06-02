"""3D visualization of the EA manifold at L40 (the steering layer).

For each projection (PCA top-3 and SAVE top-3):
  - point cloud, colored by frame_score (FS=0 conversational ... FS=6 evaluation)
  - 7 FS centroids as large markers in calendar order
  - linear chord (FS0 -> FS6) as a straight line
  - paper-spline (cubic spline through 7 centroids) as a curve
  - SAVE-arc (piecewise-linear through centroids in FS order) as a curve

Outputs interactive HTML with all three projections side-by-side.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import CubicSpline

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))
from load_acts import load_labels, read_last_token_per_seq
from save_recovery import save_recovery

SHARD = Path("/data/artifacts/rohan/manifolds/eval_awareness_olmo31_32b_v1/shard_dp00")
LABELS = Path("/home/rkathuria/manifolds/data/eval_awareness_labels.jsonl")
LAYER = 40
POSITION_FROM_END = 10  # same as steering extraction


def build_paths(centroids, K=51):
    """Build the 3 steering paths used in v6 (and SAVE-arc from v5)."""
    src, tgt = 0, 6
    # linear
    linear = np.array([(1 - t) * centroids[src] + t * centroids[tgt]
                        for t in np.linspace(0, 1, K)])
    # paper spline: cubic spline parameterized by FS index, sampled t in [src, tgt]
    sp = CubicSpline(np.arange(len(centroids)), centroids, bc_type="natural", axis=0)
    paper = sp(np.linspace(src, tgt, K))
    # SAVE-arc: piecewise-linear through centroids in FS order
    pts = centroids
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    L = cum[-1]
    save_arc = np.zeros((K, pts.shape[1]))
    for j, t in enumerate(np.linspace(0, L, K)):
        s = int(np.searchsorted(cum, t, side="right") - 1)
        s = min(s, len(pts) - 2)
        local = (t - cum[s]) / max(seg[s], 1e-12)
        save_arc[j] = (1 - local) * pts[s] + local * pts[s + 1]
    return {"linear": linear, "paper": paper, "save_arc": save_arc}


def project(X, basis):
    """Project [N, d] data onto [d, k] basis -> [N, k]."""
    return X @ basis


PCA_TITLE = "PCA (paper version)"
SAVE_TITLE = "SAVE estimation"
PATH_COLORS = {"linear": "#1f77b4", "paper": "#ff7f0e", "save_arc": "#2ca02c"}


def add_manifold(fig, row, col, proj, fs, show_colorbar, show_legend):
    """Add the FS point cloud + centroids + steering paths to one subplot scene."""
    fig.add_trace(go.Scatter3d(
        x=proj["X"][:, 0], y=proj["X"][:, 1], z=proj["X"][:, 2], mode="markers",
        marker=dict(size=2.5, color=fs, colorscale="Viridis", opacity=0.5,
                    showscale=show_colorbar,
                    colorbar=dict(title="FS", thickness=12, x=1.0, len=0.85)),
        text=[f"FS={f}" for f in fs], hovertemplate="%{text}<extra></extra>",
        name="points", showlegend=False,
    ), row=row, col=col)
    fig.add_trace(go.Scatter3d(
        x=proj["C"][:, 0], y=proj["C"][:, 1], z=proj["C"][:, 2],
        mode="markers+text",
        marker=dict(size=8, color=list(range(7)), colorscale="Viridis",
                    line=dict(color="black", width=2), opacity=1.0),
        text=[f"FS={k}" for k in range(7)], textposition="top center",
        name="centroids", showlegend=False,
    ), row=row, col=col)
    for method, p in proj["paths"].items():
        fig.add_trace(go.Scatter3d(
            x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="lines",
            line=dict(color=PATH_COLORS[method], width=6),
            name=method, legendgroup=method, showlegend=show_legend,
        ), row=row, col=col)


def main():
    out_dir = Path("/home/rkathuria/manifolds/figures/ea_3d_L40")
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(LABELS)
    X, pids = read_last_token_per_seq(SHARD, LAYER, position_from_end=POSITION_FROM_END)
    fs = np.array([labels[p]["frame_score"] for p in pids])
    print(f"loaded {X.shape[0]} activations at L{LAYER}, dim={X.shape[1]}")
    print(f"  FS dist: {np.bincount(fs)}")

    centroids = np.stack([X[fs == k].mean(0) for k in range(7)])
    Xc = X - X.mean(0, keepdims=True)
    print(f"  chord ‖c_FS6 − c_FS0‖ = {np.linalg.norm(centroids[6]-centroids[0]):.2f}")

    print("\ncomputing PCA top-3...")
    Q, S, _ = np.linalg.svd(Xc.astype(np.float64), full_matrices=False)
    pca_basis = (Xc.T @ Q[:, :3] / (S[:3] + 1e-12))[:, :3]   # [d, 3]
    pca_basis /= np.linalg.norm(pca_basis, axis=0, keepdims=True)
    pca_var = (S[:3] ** 2) / (S ** 2).sum()
    print(f"  top-3 PCA var explained: {pca_var}")
    print(f"  cumulative: {pca_var.sum():.3f}")

    print("\ncomputing SAVE top-3 (probe = FS centroid difference c_FS6 − c_FS0)...")
    probe_w = centroids[6] - centroids[0]
    probe_w /= np.linalg.norm(probe_w)
    save_res = save_recovery(X, probe_w, n_slices=10, d_M=3)
    save_basis = save_res.save_dirs       # [d, 3]
    print(f"  SAVE top eigenvalues: {save_res.eigvals[:5]}")
    eig_ratio = save_res.eigvals[0] / max(save_res.eigvals[10], 1e-12)
    print(f"  eigval[0] / eigval[10]: {eig_ratio:.2f}")

    paths = build_paths(centroids, K=51)

    proj_PCA = {
        "X": project(X, pca_basis),
        "C": project(centroids, pca_basis),
        "paths": {m: project(p, pca_basis) for m, p in paths.items()},
    }
    proj_SAVE = {
        "X": project(X, save_basis),
        "C": project(centroids, save_basis),
        "paths": {m: project(p, save_basis) for m, p in paths.items()},
    }

    # One figure, two independently-rotatable 3D scenes, title above each.
    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.02,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(PCA_TITLE, SAVE_TITLE),
    )
    add_manifold(fig, 1, 1, proj_PCA, fs, show_colorbar=False, show_legend=True)
    add_manifold(fig, 1, 2, proj_SAVE, fs, show_colorbar=True, show_legend=False)
    scene_opts = dict(xaxis_title="dim 1", yaxis_title="dim 2",
                      zaxis_title="dim 3", aspectmode="data")
    fig.update_layout(
        scene=scene_opts, scene2=scene_opts,
        width=1400, height=680, margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=0, yanchor="top"),
    )
    for ann in fig.layout.annotations:
        ann.font = dict(size=18)

    out = out_dir / f"ea_3d_L{LAYER}.html"
    fig.write_html(out, include_plotlyjs=True)   # embedded -> offline-capable
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

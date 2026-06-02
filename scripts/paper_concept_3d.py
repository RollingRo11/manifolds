"""Generalized 3D PCA-vs-SAVE manifold comparison for paper steering concepts.

Mirrors scripts/ea_3d_viz.py, but parameterized over the Goodfire-paper concept
families (weekday / month / letter / age) instead of the eval-awareness frame
score. For a chosen family it:

  - loads last-token activations for that family from a harvested shard,
  - computes a PCA top-3 basis and a SAVE top-3 basis (probe = c_last - c_first
    along the canonical concept order),
  - renders ONE clean Plotly figure with two side-by-side 3D scenes
    ("PCA (paper version)" | "SAVE estimation"), each independently rotatable,
  - writes <fam>_3d_L<layer>.html.

The single figure is the blog-embeddable deliverable: just the two 3D views and
a title above each. HTML embeds plotly.js inline (include_plotlyjs=True) so it
is fully self-contained and renders offline / off-cluster.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import CubicSpline

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))
from load_acts import load_labels, read_last_token_per_seq
from save_recovery import save_recovery

PCA_TITLE = "PCA (paper version)"
SAVE_TITLE = "SAVE estimation"

# Canonical order per family (defines the ordinal axis + the SAVE probe endpoints).
FAMILY_ORDER = {
    "weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                "Saturday", "Sunday"],
    "month": ["January", "February", "March", "April", "May", "June", "July",
              "August", "September", "October", "November", "December"],
    "letter": [chr(c) for c in range(ord("C"), ord("Z") + 1)],   # C..Z  (paper)
    "age": [str(a) for a in range(1, 100)],                       # 1..99 (paper)
}


def normalize(v):
    return v / (np.linalg.norm(v) + 1e-12)


def build_paths(centroids, K=51):
    """linear chord, paper cubic-spline, and SAVE-arc through the centroids."""
    src, tgt = 0, len(centroids) - 1
    linear = np.array([(1 - t) * centroids[src] + t * centroids[tgt]
                       for t in np.linspace(0, 1, K)])
    sp = CubicSpline(np.arange(len(centroids)), centroids, bc_type="natural", axis=0)
    paper = sp(np.linspace(src, tgt, K))
    pts = centroids
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    L = cum[-1]
    save_arc = np.zeros((K, pts.shape[1]))
    for j, t in enumerate(np.linspace(0, L, K)):
        s = min(int(np.searchsorted(cum, t, side="right") - 1), len(pts) - 2)
        local = (t - cum[s]) / max(seg[s], 1e-12)
        save_arc[j] = (1 - local) * pts[s] + local * pts[s + 1]
    return {"linear": linear, "paper": paper, "save_arc": save_arc}


def project(X, basis):
    return X @ basis


PATH_COLORS = {"linear": "#1f77b4", "paper": "#ff7f0e", "save_arc": "#2ca02c"}


def add_manifold(fig, row, col, proj, cidx, order, cbar, show_colorbar, show_legend):
    """Add the point cloud + centroids + steering paths to one subplot scene."""
    n = len(order)
    label_centroids = n <= 14   # don't clutter 99-age plots with text labels
    fig.add_trace(go.Scatter3d(
        x=proj["X"][:, 0], y=proj["X"][:, 1], z=proj["X"][:, 2], mode="markers",
        marker=dict(size=2.5, color=cidx, colorscale="Viridis", opacity=0.5,
                    showscale=show_colorbar,
                    colorbar=dict(title=cbar, thickness=12, x=1.0, len=0.85)),
        text=[f"{cbar}={order[i]}" for i in cidx],
        hovertemplate="%{text}<extra></extra>",
        name="points", showlegend=False,
    ), row=row, col=col)
    fig.add_trace(go.Scatter3d(
        x=proj["C"][:, 0], y=proj["C"][:, 1], z=proj["C"][:, 2],
        mode="markers+text" if label_centroids else "markers",
        marker=dict(size=8, color=list(range(n)), colorscale="Viridis",
                    line=dict(color="black", width=2), opacity=1.0),
        text=[str(o) for o in order] if label_centroids else None,
        textposition="top center", name="centroids", showlegend=False,
    ), row=row, col=col)
    for method, p in proj["paths"].items():
        fig.add_trace(go.Scatter3d(
            x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="lines",
            line=dict(color=PATH_COLORS[method], width=6),
            name=method, legendgroup=method, showlegend=show_legend,
        ), row=row, col=col)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=sorted(FAMILY_ORDER))
    ap.add_argument("--shard", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--layer", type=int, default=40)
    ap.add_argument("--position-from-end", type=int, default=0)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    order = FAMILY_ORDER[args.family]
    idx_of = {c: i for i, c in enumerate(order)}
    cbar = {"weekday": "day", "month": "month", "letter": "letter",
            "age": "age"}[args.family]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    labels = load_labels(args.labels)
    X, pids = read_last_token_per_seq(args.shard, args.layer,
                                      position_from_end=args.position_from_end)
    keep = [i for i, p in enumerate(pids)
            if labels[p].get("family") == args.family
            and str(labels[p].get("concept")) in idx_of]
    if not keep:
        raise SystemExit(f"no rows for family={args.family!r} in {args.labels}")
    X = X[keep].astype(np.float64)
    cidx = np.array([idx_of[str(labels[pids[i]]["concept"])] for i in keep])
    print(f"loaded {X.shape[0]} '{args.family}' acts at L{args.layer}, dim={X.shape[1]}")

    # Centroids in canonical order (skip any concept with no samples), with a
    # dense color remap so the colorscale spans exactly the present concepts.
    used_order, centroids = [], []
    for k in range(len(order)):
        m = cidx == k
        if m.sum() == 0:
            continue
        used_order.append(order[k])
        centroids.append(X[m].mean(0))
    centroids = np.stack(centroids)
    remap = {idx_of[o]: j for j, o in enumerate(used_order)}
    cidx_dense = np.array([remap[c] for c in cidx])
    print(f"  {len(used_order)}/{len(order)} concepts present")

    Xc = X - X.mean(0, keepdims=True)
    print(f"  chord ||c_last - c_first|| = {np.linalg.norm(centroids[-1]-centroids[0]):.2f}")

    print("computing PCA top-3...")
    Q, S, _ = np.linalg.svd(Xc, full_matrices=False)
    pca_basis = (Xc.T @ Q[:, :3] / (S[:3] + 1e-12))[:, :3]
    pca_basis /= np.linalg.norm(pca_basis, axis=0, keepdims=True)
    pca_var = (S[:3] ** 2) / (S ** 2).sum()
    print(f"  top-3 PCA var explained: {pca_var}  cum={pca_var.sum():.3f}")

    print("computing SAVE top-3 (probe = c_last - c_first)...")
    probe_w = normalize(centroids[-1] - centroids[0])
    save_res = save_recovery(X, probe_w, n_slices=10, d_M=3)
    save_basis = save_res.save_dirs
    print(f"  SAVE top eigenvalues: {save_res.eigvals[:5]}")

    paths = build_paths(centroids, K=51)
    proj = lambda B: dict(X=project(X, B), C=project(centroids, B),
                          paths={m: project(p, B) for m, p in paths.items()})
    P, Sv = proj(pca_basis), proj(save_basis)

    # One figure, two independently-rotatable 3D scenes, title above each.
    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.02,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(PCA_TITLE, SAVE_TITLE),
    )
    add_manifold(fig, 1, 1, P, cidx_dense, used_order, cbar,
                 show_colorbar=False, show_legend=True)
    add_manifold(fig, 1, 2, Sv, cidx_dense, used_order, cbar,
                 show_colorbar=True, show_legend=False)
    scene_opts = dict(xaxis_title="dim 1", yaxis_title="dim 2",
                      zaxis_title="dim 3", aspectmode="data")
    fig.update_layout(
        scene=scene_opts, scene2=scene_opts,
        width=1400, height=680, margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=0,
                    yanchor="top"),
    )
    # Make the two subplot titles a touch larger / cleaner for a blog.
    for ann in fig.layout.annotations:
        ann.font = dict(size=18)

    out = args.out_dir / f"{args.family}_3d_L{args.layer}.html"
    fig.write_html(out, include_plotlyjs=True)   # embedded -> offline-capable
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

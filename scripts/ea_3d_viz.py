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


def make_3d_plot(coords, fs, centroid_coords, paths_3d, title):
    """Single 3D scatter+lines figure."""
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        mode="markers",
        marker=dict(size=2.5, color=fs, colorscale="Viridis", opacity=0.45,
                     showscale=True, colorbar=dict(title="FS", thickness=10)),
        text=[f"FS={f}" for f in fs], hovertemplate="%{text}",
        name="points",
    ))

    fig.add_trace(go.Scatter3d(
        x=centroid_coords[:, 0], y=centroid_coords[:, 1], z=centroid_coords[:, 2],
        mode="markers+text",
        marker=dict(size=10, color=list(range(7)), colorscale="Viridis",
                     line=dict(color="black", width=2), opacity=1.0),
        text=[f"FS={k}" for k in range(7)],
        textposition="top center",
        name="centroids",
    ))

    colors = {"linear": "#1f77b4", "paper": "#ff7f0e", "save_arc": "#2ca02c"}
    for method, p in paths_3d.items():
        fig.add_trace(go.Scatter3d(
            x=p[:, 0], y=p[:, 1], z=p[:, 2],
            mode="lines",
            line=dict(color=colors[method], width=6),
            name=method,
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="dim 1", yaxis_title="dim 2", zaxis_title="dim 3",
            aspectmode="data",
        ),
        width=900, height=720,
        legend=dict(x=0.02, y=0.95),
    )
    return fig


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

    fig_pca = make_3d_plot(proj_PCA["X"], fs, proj_PCA["C"], proj_PCA["paths"],
                            f"PCA top-3 (var={pca_var.sum():.2%}) — L{LAYER} EA")
    fig_save = make_3d_plot(proj_SAVE["X"], fs, proj_SAVE["C"], proj_SAVE["paths"],
                             f"SAVE top-3 (probe=c_FS6−c_FS0) — L{LAYER} EA")

    pca_path = out_dir / f"ea_3d_L{LAYER}_pca.html"
    save_path = out_dir / f"ea_3d_L{LAYER}_save.html"
    # include_plotlyjs=True embeds plotly.js inline -> self-contained / offline.
    fig_pca.write_html(pca_path, include_plotlyjs=True)
    fig_save.write_html(save_path, include_plotlyjs=True)
    print(f"\nwrote {pca_path}")
    print(f"wrote {save_path}")

    combined_path = out_dir / f"ea_3d_L{LAYER}_combined.html"
    pca_html = fig_pca.to_html(include_plotlyjs=True, full_html=False)
    save_html = fig_save.to_html(include_plotlyjs=False, full_html=False)
    combined_path.write_text(f"""<!doctype html><html><head>
<title>EA 3D manifold L{LAYER}</title>
<style>body {{ font-family: sans-serif; margin: 20px; }} h2 {{ margin-top: 30px; }}</style>
</head><body>
<h1>EA manifold visualization — OLMo-3.1-Think-32B, L{LAYER}, position_from_end={POSITION_FROM_END}</h1>
<p><b>Setup:</b> {X.shape[0]} EA prompts, FS=0 (conversational) → FS=6 (evaluation), 7 centroids.
Three steering paths overlaid: <span style='color:#1f77b4'><b>linear</b></span> chord,
<span style='color:#ff7f0e'><b>paper-spline</b></span> (cubic spline through 7 centroids),
<span style='color:#2ca02c'><b>SAVE-arc</b></span> (piecewise-linear through centroids in FS order).</p>
<h2>SAVE top-3 projection</h2>
<p>SAVE basis is anchored on the FS-difference probe direction. This is the basis we extract
the manifold structure from. If the manifold is meaningful, FS gradient should be visible
as a clean trajectory.</p>
{save_html}
<h2>PCA top-3 projection</h2>
<p>PCA captures the largest ambient variance directions. The FS structure may be obscured
by other variance (prompt content, position effects).</p>
{pca_html}
</body></html>""")
    print(f"wrote {combined_path}")


if __name__ == "__main__":
    main()

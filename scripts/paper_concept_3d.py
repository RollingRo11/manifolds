"""Generalized 3D PCA-vs-SAVE manifold comparison for paper steering concepts.

Mirrors scripts/ea_3d_viz.py, but parameterized over the Goodfire-paper concept
families (weekday / month / letter / age) instead of the eval-awareness frame
score. For a chosen family it:

  - loads last-token activations for that family from a harvested shard,
  - computes a PCA top-3 basis and a SAVE top-3 basis (probe = c_last - c_first
    along the canonical concept order),
  - overlays the per-concept centroids and the linear / paper-spline / SAVE-arc
    steering paths,
  - writes <fam>_pca.html, <fam>_save.html and <fam>_combined.html.

IMPORTANT: HTML is written with include_plotlyjs=True (plotly.js embedded inline)
so the files are fully self-contained and render offline / off-cluster — no
cdn.plot.ly dependency.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import CubicSpline

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))
from load_acts import load_labels, read_last_token_per_seq
from save_recovery import save_recovery

# Canonical order per family (defines the ordinal axis + the SAVE probe endpoints).
FAMILY_ORDER = {
    "weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                "Saturday", "Sunday"],
    "month": ["January", "February", "March", "April", "May", "June", "July",
              "August", "September", "October", "November", "December"],
    "letter": [chr(c) for c in range(ord("C"), ord("Z") + 1)],   # C..Z (paper)
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


def make_3d_plot(coords, cidx, centroid_coords, paths_3d, title, order, cbar_title):
    fig = go.Figure()
    n = len(order)
    label_centroids = n <= 14  # don't clutter 99-age plots with text labels

    fig.add_trace(go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2], mode="markers",
        marker=dict(size=2.5, color=cidx, colorscale="Viridis", opacity=0.45,
                    showscale=True, colorbar=dict(title=cbar_title, thickness=10)),
        text=[f"{cbar_title}={order[i]}" for i in cidx], hovertemplate="%{text}",
        name="points",
    ))
    fig.add_trace(go.Scatter3d(
        x=centroid_coords[:, 0], y=centroid_coords[:, 1], z=centroid_coords[:, 2],
        mode="markers+text" if label_centroids else "markers",
        marker=dict(size=9, color=list(range(n)), colorscale="Viridis",
                    line=dict(color="black", width=2), opacity=1.0),
        text=[str(o) for o in order] if label_centroids else None,
        textposition="top center", name="centroids",
    ))
    colors = {"linear": "#1f77b4", "paper": "#ff7f0e", "save_arc": "#2ca02c"}
    for method, p in paths_3d.items():
        fig.add_trace(go.Scatter3d(
            x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="lines",
            line=dict(color=colors[method], width=6), name=method,
        ))
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="dim 1", yaxis_title="dim 2", zaxis_title="dim 3",
                   aspectmode="data"),
        width=900, height=720, legend=dict(x=0.02, y=0.95),
    )
    return fig


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
    present = sorted(set(cidx.tolist()))
    print(f"loaded {X.shape[0]} '{args.family}' acts at L{args.layer}, dim={X.shape[1]}")
    print(f"  {len(present)}/{len(order)} concepts present")

    # Centroids in canonical order (skip any concept with no samples).
    used_order, centroids = [], []
    for k in range(len(order)):
        m = cidx == k
        if m.sum() == 0:
            continue
        used_order.append(order[k])
        centroids.append(X[m].mean(0))
    centroids = np.stack(centroids)
    # remap colors to dense 0..len(used_order)-1 so colorscale spans the data
    remap = {idx_of[o]: j for j, o in enumerate(used_order)}
    cidx_dense = np.array([remap[c] for c in cidx])

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

    fam_label = args.family.capitalize()
    fig_pca = make_3d_plot(P["X"], cidx_dense, P["C"], P["paths"],
                           f"PCA top-3 (var={pca_var.sum():.2%}) — L{args.layer} {fam_label}",
                           used_order, cbar)
    fig_save = make_3d_plot(Sv["X"], cidx_dense, Sv["C"], Sv["paths"],
                            f"SAVE top-3 (probe=c_last−c_first) — L{args.layer} {fam_label}",
                            used_order, cbar)

    pca_path = args.out_dir / f"{args.family}_3d_L{args.layer}_pca.html"
    save_path = args.out_dir / f"{args.family}_3d_L{args.layer}_save.html"
    # include_plotlyjs=True embeds plotly.js inline -> fully offline-capable.
    fig_pca.write_html(pca_path, include_plotlyjs=True)
    fig_save.write_html(save_path, include_plotlyjs=True)
    print(f"wrote {pca_path}")
    print(f"wrote {save_path}")

    combined_path = args.out_dir / f"{args.family}_3d_L{args.layer}_combined.html"
    pca_html = fig_pca.to_html(include_plotlyjs=True, full_html=False)
    save_html = fig_save.to_html(include_plotlyjs=False, full_html=False)
    combined_path.write_text(f"""<!doctype html><html><head>
<meta charset="utf-8"/><title>{fam_label} 3D manifold L{args.layer}</title>
<style>body {{ font-family: sans-serif; margin: 20px; }} h2 {{ margin-top: 30px; }}</style>
</head><body>
<h1>{fam_label} manifold — PCA vs SAVE — OLMo-3.1-Think-32B, L{args.layer}</h1>
<p><b>Setup:</b> {X.shape[0]} '{args.family}' prompts, {len(used_order)} concepts
({used_order[0]} → {used_order[-1]}). Paths overlaid:
<span style='color:#1f77b4'><b>linear</b></span> chord,
<span style='color:#ff7f0e'><b>paper-spline</b></span>,
<span style='color:#2ca02c'><b>SAVE-arc</b></span>. Self-contained (plotly.js embedded).</p>
<h2>PCA top-3</h2>
{pca_html}
<h2>SAVE top-3</h2>
{save_html}
</body></html>""")
    print(f"wrote {combined_path}")


if __name__ == "__main__":
    main()

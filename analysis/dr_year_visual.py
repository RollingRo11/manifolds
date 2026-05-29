"""Visual test of DR on the year manifold (helix).

Generates an interactive plotly HTML with three 3-D scenes side by side:
  - supervised PCA (paper-style ground truth)
  - SAVE recovery (pure SAVE — should look 2-D-flat)
  - DR recovery (combined SAVE + SIR — should recover the helical 3rd dim)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent))
from dr_recovery import dr_recovery
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


def add_3d_scene(fig, row, col, points, values, centroid_pts, centroid_values,
                  axis_titles, scene_title):
    fig.add_trace(go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode="markers",
        marker=dict(size=2, color=values, colorscale="Viridis", opacity=0.5),
        text=[f"value={v}" for v in values], hoverinfo="text",
        showlegend=False, name="prompts",
    ), row=row, col=col)
    fig.add_trace(go.Scatter3d(
        x=centroid_pts[:, 0], y=centroid_pts[:, 1], z=centroid_pts[:, 2],
        mode="lines+markers",
        line=dict(color="black", width=4),
        marker=dict(size=4, color=centroid_values, colorscale="Viridis",
                    line=dict(color="black", width=1)),
        hoverinfo="text", text=[f"value={v}" for v in centroid_values],
        showlegend=False,
    ), row=row, col=col)


def main():
    layer = 32
    fam = "year"
    labels = load_labels(Path("/home/rkathuria/manifolds/data/labels_v4_paper.jsonl"))
    X_full, pids = read_last_token_per_seq(
        Path("/data/artifacts/rohan/manifolds/paper_olmo31_32b_v4/shard_dp00"), layer)
    keep = [i for i, p in enumerate(pids) if labels[p]["family"] == fam]
    X = X_full[keep]
    values = [int(labels[pids[i]]["value"]) for i in keep]

    # Supervised PCA top-3
    sup_emb = PCA(n_components=3).fit_transform(X)
    uniq_s, sup_cents = per_value_centroids(sup_emb, values)

    # Probe + SAVE + DR
    w = chord_probe_extremes(X, values)
    sv = save_recovery(X, w, n_slices=10, d_M=2)
    dr = dr_recovery(X, w, n_slices=10, d_M=2)
    uniq_v, save_cents = per_value_centroids(sv.embedding, values)
    _, dr_cents = per_value_centroids(dr.embedding, values)

    print(f"\n=== {fam} layer {layer} ===")
    print(f"  SAVE top-5 eigvals: {[round(float(x),2) for x in sv.eigvals[:5]]}")
    print(f"  DR   top-5 eigvals: {[round(float(x),2) for x in dr.eigvals[:5]]}")
    print(f"    pure SAVE eigvals: {[round(float(x),2) for x in dr.save_eigvals[:5]]}")
    print(f"    pure SIR  eigvals: {[round(float(x),2) for x in dr.sir_eigvals[:5]]}")
    print(f"    c_balance = {dr.c_balance:.4f}")

    from scipy.spatial import procrustes
    _, _, save_proc = procrustes(sup_cents, save_cents)
    _, _, dr_proc = procrustes(sup_cents, dr_cents)
    print(f"  SAVE Procrustes vs supervised: {save_proc:.4f}")
    print(f"  DR   Procrustes vs supervised: {dr_proc:.4f}")

    # Interactive 3-panel plot
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "scene"}] * 3],
        subplot_titles=(
            f"<b>SUPERVISED PCA</b>",
            f"<b>SAVE</b><br><span style='font-size:10px'>"
            f"top-5 eigvals = {[round(float(x),1) for x in sv.eigvals[:5]]}<br>"
            f"Procrustes = {save_proc:.3f}</span>",
            f"<b>DR (SAVE + SIR)</b><br><span style='font-size:10px'>"
            f"top-5 eigvals = {[round(float(x),1) for x in dr.eigvals[:5]]}<br>"
            f"Procrustes = {dr_proc:.3f}</span>",
        ),
    )

    add_3d_scene(fig, 1, 1, sup_emb, values, sup_cents, uniq_s,
                  ["PC1", "PC2", "PC3"], "supervised PCA")
    add_3d_scene(fig, 1, 2, sv.embedding, values, save_cents, uniq_v,
                  ["s", "save_dir_1", "save_dir_2"], "SAVE")
    add_3d_scene(fig, 1, 3, dr.embedding, values, dr_cents, uniq_v,
                  ["s", "dr_dir_1", "dr_dir_2"], "DR")

    fig.update_layout(
        title=f"DR vs SAVE vs supervised — year manifold (helix), layer {layer}",
        height=750, width=1900,
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
        scene2=dict(xaxis_title="s = ŵ·h", yaxis_title="save_dir_1", zaxis_title="save_dir_2"),
        scene3=dict(xaxis_title="s = ŵ·h", yaxis_title="dr_dir_1", zaxis_title="dr_dir_2"),
        margin=dict(l=0, r=0, t=100, b=0),
    )
    out = Path("/home/rkathuria/manifolds/figures/interactive/year_dr.html")
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

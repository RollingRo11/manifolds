"""DR vs SAVE vs supervised — all paper manifolds, multiple layers.

Generates one interactive HTML per (manifold, layer) showing three 3-D scenes
side by side. Diagnostic: does DR add visible 3-D depth where SAVE was flat?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial import procrustes
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


def add_scene(fig, row, col, points, values, cents, cents_v):
    fig.add_trace(go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode="markers",
        marker=dict(size=2, color=values, colorscale="Viridis", opacity=0.5),
        text=[f"{v}" for v in values], hoverinfo="text", showlegend=False,
    ), row=row, col=col)
    fig.add_trace(go.Scatter3d(
        x=cents[:, 0], y=cents[:, 1], z=cents[:, 2],
        mode="lines+markers",
        line=dict(color="black", width=4),
        marker=dict(size=4, color=cents_v, colorscale="Viridis",
                    line=dict(color="black", width=1)),
        text=[f"{v}" for v in cents_v], hoverinfo="text", showlegend=False,
    ), row=row, col=col)


def render_one(fam, layer, X, values, out_path):
    sup_emb = PCA(n_components=3).fit_transform(X)
    uniq, sup_cents = per_value_centroids(sup_emb, values)

    w = chord_probe_extremes(X, values)
    sv = save_recovery(X, w, n_slices=10, d_M=2)
    dr = dr_recovery(X, w, n_slices=10, d_M=2)
    _, save_cents = per_value_centroids(sv.embedding, values)
    _, dr_cents = per_value_centroids(dr.embedding, values)

    _, _, save_p = procrustes(sup_cents, save_cents)
    _, _, dr_p = procrustes(sup_cents, dr_cents)

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "scene"}] * 3],
        subplot_titles=(
            f"<b>SUPERVISED PCA</b>",
            f"<b>SAVE</b><br><span style='font-size:10px'>"
            f"eigs = {[round(float(x),1) for x in sv.eigvals[:4]]}<br>"
            f"proc = {save_p:.3f}</span>",
            f"<b>DR</b> (SAVE + SIR)<br><span style='font-size:10px'>"
            f"DR eigs = {[round(float(x),1) for x in dr.eigvals[:4]]}<br>"
            f"SAVE: {[round(float(x),1) for x in dr.save_eigvals[:3]]}  "
            f"SIR: {[round(float(x),2) for x in dr.sir_eigvals[:3]]}<br>"
            f"proc = {dr_p:.3f}  c={dr.c_balance:.2f}</span>",
        ),
    )
    add_scene(fig, 1, 1, sup_emb, values, sup_cents, uniq)
    add_scene(fig, 1, 2, sv.embedding, values, save_cents, uniq)
    add_scene(fig, 1, 3, dr.embedding, values, dr_cents, uniq)
    fig.update_layout(
        title=f"DR vs SAVE vs supervised — {fam} manifold, layer {layer}",
        height=750, width=1900,
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
        scene2=dict(xaxis_title="s", yaxis_title="save_1", zaxis_title="save_2"),
        scene3=dict(xaxis_title="s", yaxis_title="dr_1", zaxis_title="dr_2"),
        margin=dict(l=0, r=0, t=100, b=0),
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    return {"save_proc": save_p, "dr_proc": dr_p,
            "save_top4_eigs": [float(x) for x in sv.eigvals[:4]],
            "dr_top4_eigs": [float(x) for x in dr.eigvals[:4]],
            "save_eigs_in_dr": [float(x) for x in dr.save_eigvals[:4]],
            "sir_eigs_in_dr": [float(x) for x in dr.sir_eigvals[:4]],
            "c_balance": dr.c_balance}


def main():
    out_dir = Path("/home/rkathuria/manifolds/figures/interactive/dr")
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_v4 = load_labels(Path("/home/rkathuria/manifolds/data/labels_v4_paper.jsonl"))
    labels_v3 = load_labels(Path("/home/rkathuria/manifolds/data/labels_v3.jsonl"))
    weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

    # Run for the most informative layers per manifold
    targets = [
        ("year",        16, "v4"),
        ("year",        24, "v4"),
        ("year",        32, "v4"),
        ("year",        40, "v4"),
        ("temperature", 16, "v4"),
        ("age",         16, "v4"),
        ("days",        16, "v3"),
        ("days",        24, "v3"),
    ]

    summaries = []
    for fam, layer, source in targets:
        if source == "v4":
            shard = "/data/artifacts/rohan/manifolds/paper_olmo31_32b_v4/shard_dp00"
            X_full, pids = read_last_token_per_seq(Path(shard), layer)
            keep = [i for i, p in enumerate(pids) if labels_v4[p]["family"] == fam]
            X = X_full[keep]
            values = [int(labels_v4[pids[i]]["value"]) for i in keep]
        else:
            shard = "/data/artifacts/rohan/manifolds/concepts_olmo31_32b_v3/shard_dp00"
            X_full, pids = read_last_token_per_seq(Path(shard), layer)
            keep = [i for i, p in enumerate(pids) if labels_v3[p]["family"] == "weekday"]
            X = X_full[keep]
            values = [weekday_order.index(labels_v3[pids[i]]["concept"]) for i in keep]

        out_path = out_dir / f"{fam}_L{layer:02d}.html"
        s = render_one(fam, layer, X, values, out_path)
        s["fam"] = fam; s["layer"] = layer; s["n"] = X.shape[0]
        summaries.append(s)
        print(f"[{fam:>11} L{layer:02d}] save_proc={s['save_proc']:.4f}  dr_proc={s['dr_proc']:.4f}  "
              f"c={s['c_balance']:.2f}  wrote {out_path.name}")

    # Tiny index page
    idx = "<!doctype html><html><body><h1>DR vs SAVE interactive</h1><ul>"
    for s in summaries:
        idx += (f"<li><a href='{s['fam']}_L{s['layer']:02d}.html'>"
                f"{s['fam']} L{s['layer']}</a> — "
                f"SAVE proc {s['save_proc']:.3f}, DR proc {s['dr_proc']:.3f}, "
                f"c={s['c_balance']:.1f}</li>")
    idx += "</ul></body></html>"
    (out_dir / "index.html").write_text(idx)
    print(f"\nindex: {out_dir}/index.html")


if __name__ == "__main__":
    main()

"""Interactive 3D plots (plotly) — supervised PCA vs SAVE recovery side by side.

Outputs one HTML per manifold (and one combined HTML) you can open in a browser
and rotate / zoom freely.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent))
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


def make_interactive(fam: str, geometry: str, X: np.ndarray, values: list,
                     out_path: Path, layer: int):
    sup_emb = PCA(n_components=3).fit_transform(X)
    uniq_s, sup_cents = per_value_centroids(sup_emb, values)

    w = chord_probe_extremes(X, values)
    sv = save_recovery(X, w, n_slices=10, d_M=2)
    uniq_v, save_cents = per_value_centroids(sv.embedding, values)

    # Build figure with two 3D scenes
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(
            f"<b>SUPERVISED PCA</b> (paper-style ground truth)<br>"
            f"<span style='font-size:11px'>{fam} — {geometry}, n={X.shape[0]}, layer {layer}</span>",
            f"<b>SAVE RECOVERED</b> (single chord probe)<br>"
            f"<span style='font-size:11px'>top-eigvals = "
            f"{[round(float(x),2) for x in sv.eigvals[:3]]}</span>",
        ),
    )

    # Per-prompt scatter (LEFT)
    fig.add_trace(go.Scatter3d(
        x=sup_emb[:, 0], y=sup_emb[:, 1], z=sup_emb[:, 2],
        mode="markers",
        marker=dict(size=3, color=values, colorscale="Viridis",
                    colorbar=dict(title=fam, x=0.45, len=0.7), opacity=0.55),
        text=[f"{fam}={v}" for v in values],
        hoverinfo="text", showlegend=False, name="prompts",
    ), row=1, col=1)
    # Centroid spline (LEFT)
    fig.add_trace(go.Scatter3d(
        x=sup_cents[:, 0], y=sup_cents[:, 1], z=sup_cents[:, 2],
        mode="lines+markers",
        line=dict(color="black", width=4),
        marker=dict(size=8, color=uniq_s, colorscale="Viridis",
                    line=dict(color="black", width=1)),
        hoverinfo="text", text=[f"{fam}={v}" for v in uniq_s],
        showlegend=False, name="centroids",
    ), row=1, col=1)

    # Per-prompt scatter (RIGHT)
    fig.add_trace(go.Scatter3d(
        x=sv.embedding[:, 0], y=sv.embedding[:, 1], z=sv.embedding[:, 2],
        mode="markers",
        marker=dict(size=3, color=values, colorscale="Viridis",
                    showscale=False, opacity=0.55),
        text=[f"{fam}={v}" for v in values],
        hoverinfo="text", showlegend=False,
    ), row=1, col=2)
    # Centroid spline (RIGHT)
    fig.add_trace(go.Scatter3d(
        x=save_cents[:, 0], y=save_cents[:, 1], z=save_cents[:, 2],
        mode="lines+markers",
        line=dict(color="black", width=4),
        marker=dict(size=8, color=uniq_v, colorscale="Viridis",
                    showscale=False, line=dict(color="black", width=1)),
        hoverinfo="text", text=[f"{fam}={v}" for v in uniq_v],
        showlegend=False,
    ), row=1, col=2)

    fig.update_layout(
        title=f"Manifold recovery: {fam} ({geometry}) — supervised vs SAVE",
        height=650, width=1500,
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
        scene2=dict(xaxis_title="s = ŵ·h", yaxis_title="SAVE dir 1",
                    zaxis_title="SAVE dir 2"),
        margin=dict(l=0, r=0, t=80, b=0),
    )

    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"wrote {out_path}")


def make_index(out_dir: Path, families: list[str]):
    """Tiny HTML index linking to all manifolds."""
    body = "\n".join(
        f'<li><a href="{fam}.html">{fam}</a></li>' for fam in families
    )
    html = f"""<!doctype html>
<html><head><title>Interactive manifolds</title></head>
<body>
<h1>OLMo-3.1-32B-Think  layer 16  manifold recoveries</h1>
<p>Open each manifold's HTML; left = supervised PCA (truth), right = SAVE recovered from a single probe.</p>
<ul>
{body}
</ul>
</body></html>"""
    (out_dir / "index.html").write_text(html)


def main():
    layer = 16
    out_dir = Path("/home/rkathuria/manifolds/figures/interactive")
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_v4 = __import__("load_acts").load_labels(
        Path("/home/rkathuria/manifolds/data/labels_v4_paper.jsonl"))
    X4_full, pids4 = read_last_token_per_seq(
        Path("/data/artifacts/rohan/manifolds/paper_olmo31_32b_v4/shard_dp00"), layer)

    labels_v3 = __import__("load_acts").load_labels(
        Path("/home/rkathuria/manifolds/data/labels_v3.jsonl"))
    X3_full, pids3 = read_last_token_per_seq(
        Path("/data/artifacts/rohan/manifolds/concepts_olmo31_32b_v3/shard_dp00"), layer)
    weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

    families = []
    for fam, geometry in [("temperature", "line"), ("age", "line"), ("year", "helix")]:
        keep = [i for i, p in enumerate(pids4) if labels_v4[p]["family"] == fam]
        X = X4_full[keep]
        values = [int(labels_v4[pids4[i]]["value"]) for i in keep]
        make_interactive(fam, geometry, X, values, out_dir / f"{fam}.html", layer)
        families.append(fam)

    keep = [i for i, p in enumerate(pids3) if labels_v3[p]["family"] == "weekday"]
    Xw = X3_full[keep]
    valuesw = [weekday_order.index(labels_v3[pids3[i]]["concept"]) for i in keep]
    make_interactive("days", "circle", Xw, valuesw, out_dir / "days.html", layer)
    families.append("days")

    make_index(out_dir, families)
    print(f"\nopen any of:")
    for fam in families:
        print(f"  {out_dir}/{fam}.html")
    print(f"  {out_dir}/index.html")


if __name__ == "__main__":
    main()

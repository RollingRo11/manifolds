"""Eigenvalue-ratio bin selection for the paper concept manifolds.

Claim being illustrated: the number of distinct concept states ("bins") a SAVE
manifold supports is read straight off its eigenvalue spectrum — the largest
ratio drop λ[i]/λ[i+1] marks where signal eigenvalues stop and the noise floor
begins, so the recovered bin count is (i+1). No K is supplied to the method.

For a chosen family this:
  - loads last-token activations (same shard/probe as paper_concept_3d.py),
  - runs SAVE (probe = c_last - c_first) and takes the full eigenvalue spectrum,
  - finds the largest ratio drop,
  - renders ONE Plotly figure with two panels — the spectrum (log bars) and the
    ratios λ[i]/λ[i+1], with the recovered drop highlighted — and exports it as
    a static PNG + SVG (lightweight, blog-embeddable).

Static export uses kaleido (no interactivity), matching the Plotly look of the
3D figures.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore", category=DeprecationWarning)  # kaleido<1.0 notice

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))
from load_acts import load_labels, read_last_token_per_seq
from save_recovery import save_recovery

# Reuse the canonical order + true bin counts from the 3D script.
from paper_concept_3d import FAMILY_ORDER


def normalize(v):
    return v / (np.linalg.norm(v) + 1e-12)


def biggest_drop(eigvals, k_max):
    """Largest λ[i]/λ[i+1] ratio among the top k_max eigenvalues."""
    k = min(k_max, len(eigvals) - 1)
    ratios = [eigvals[i] / max(eigvals[i + 1], 1e-12) for i in range(k)]
    best = int(np.argmax(ratios))
    return best, ratios[best], np.array(ratios)


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
    fam_label = args.family.capitalize()

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
    centroids = np.stack([X[cidx == k].mean(0) for k in present])
    true_K = len(present)
    probe_w = normalize(centroids[-1] - centroids[0])
    res = save_recovery(X, probe_w, n_slices=10, d_M=3)
    eig = res.eigvals

    k_max = int(min(len(eig) - 1, max(15, true_K + 5)))
    drop_idx, drop_ratio, ratios = biggest_drop(eig, k_max)
    recovered = drop_idx + 1
    print(f"{fam_label}: N={X.shape[0]}  true bins={true_K}  "
          f"largest drop λ[{drop_idx}]/λ[{drop_idx+1}]={drop_ratio:.2f}  "
          f"-> recovered bins={recovered}")

    n_show = k_max
    base, hi = "#4c78a8", "#e45756"

    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.12,
        subplot_titles=("SAVE eigenvalue spectrum",
                        "Eigenvalue ratios  λ[i] / λ[i+1]"),
    )
    # Left: eigenvalue spectrum (log), signal bars highlighted up to the drop.
    spec_colors = [hi if i <= drop_idx else base for i in range(n_show)]
    fig.add_trace(go.Bar(x=list(range(n_show)), y=eig[:n_show],
                         marker_color=spec_colors, showlegend=False,
                         hovertemplate="λ[%{x}]=%{y:.1f}<extra></extra>"),
                  row=1, col=1)
    # Right: ratios, with the max-drop bar highlighted.
    rat_colors = [hi if i == drop_idx else base for i in range(len(ratios))]
    fig.add_trace(go.Bar(x=list(range(len(ratios))), y=ratios,
                         marker_color=rat_colors, showlegend=False,
                         hovertemplate="λ[%{x}]/λ[%{x}+1]=%{y:.2f}<extra></extra>"),
                  row=1, col=2)
    fig.add_annotation(
        row=1, col=2, x=drop_idx, y=drop_ratio,
        text=f"max drop at i={drop_idx}<br>{drop_ratio:.2f}×  →  {recovered} bins",
        showarrow=True, arrowhead=2, ax=40, ay=-40,
        font=dict(size=13, color=hi), align="left",
    )
    fig.update_yaxes(type="log", title_text="SAVE eigenvalue", row=1, col=1)
    fig.update_xaxes(title_text="eigenvector index  i", row=1, col=1)
    fig.update_yaxes(title_text="ratio λ[i] / λ[i+1]", row=1, col=2)
    fig.update_xaxes(title_text="index  i", row=1, col=2)
    fig.update_layout(
        title=dict(
            text=f"{fam_label}: eigenvalue-ratio bin selection — "
                 f"recovered {recovered} bins (true {true_K}), no K supplied",
            x=0.5, xanchor="center", font=dict(size=16)),
        width=1150, height=460, margin=dict(l=70, r=30, t=80, b=55),
        plot_bgcolor="white", bargap=0.15,
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#eee")
    for ann in fig.layout.annotations[:2]:
        ann.font = dict(size=14)

    stem = args.out_dir / f"{args.family}_eigratio_L{args.layer}"
    fig.write_image(f"{stem}.png", scale=2)
    fig.write_image(f"{stem}.svg")
    np.savez(f"{stem}.npz", eigvals=eig, ratios=ratios,
             drop_idx=drop_idx, recovered=recovered, true_K=true_K)
    print(f"wrote {stem}.png / .svg / .npz")


if __name__ == "__main__":
    main()

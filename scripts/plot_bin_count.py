"""Build the bin-count figures from the saved sweep results (figures/bin_count_sweep/
bin_count_results.json). Reads a small JSON only — no activation loading — and
exports static Plotly PNG+SVG.

Two figures:
  1. eigengap_bins  : Laplacian eigenvalue spectrum for days(L16) & months(L16);
                      the first K eigenvalues sit at ~0, then a sharp gap → K bins.
  2. layer_robustness : eigengap-recovered K and ARI@K vs layer, showing the
                      recovery is clean at L16 and degrades in deeper layers.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore", category=DeprecationWarning)

DIR = Path("/home/rkathuria/manifolds/figures/bin_count_sweep")
BASE, HI, GREY = "#4c78a8", "#e45756", "#b0b0b0"
RES = json.load(open(DIR / "bin_count_results.json"))


def row(fam, layer):
    return next(r for r in RES if r["family"] == fam and r["layer"] == layer)


def eigengap_fig():
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=("Days — layer 16", "Months — layer 16"))
    for col, fam in enumerate(["weekday", "month"], start=1):
        r = row(fam, 16)
        K = r["true_K"]
        ev = np.array(r["laplacian_eigs"])
        n = K + 5
        x = list(range(n))
        y = ev[:n]
        colors = [HI if i < K else BASE for i in range(n)]
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers+lines",
            marker=dict(size=9, color=colors, line=dict(width=0)),
            line=dict(color="#cccccc", width=1), showlegend=False,
            hovertemplate="λ[%{x}]=%{y:.4f}<extra></extra>"), row=1, col=col)
        # gap marker between index K-1 and K
        fig.add_vline(x=K - 0.5, line=dict(color=HI, dash="dash", width=2),
                      row=1, col=col)
        gap = y[K] - y[K - 1]
        fig.add_annotation(row=1, col=col, x=K - 0.5, y=y[K] * 0.6,
                           text=f"eigengap → {K} bins<br>(ARI {r['ARI_at_trueK']:.2f})",
                           showarrow=True, arrowhead=2, ax=55, ay=-25,
                           font=dict(size=13, color=HI), align="left")
        fig.update_xaxes(title_text="eigenvalue index  i", row=1, col=col,
                         dtick=2, showgrid=False)
        fig.update_yaxes(title_text="Laplacian eigenvalue λ", row=1, col=col,
                         gridcolor="#eee")
    fig.update_layout(
        title=dict(text="Number of bins = Laplacian eigengap "
                        "(K near-zero eigenvalues, then a jump) — no K supplied",
                   x=0.5, xanchor="center", font=dict(size=16)),
        width=1150, height=460, margin=dict(l=70, r=30, t=80, b=55),
        plot_bgcolor="white")
    for a in fig.layout.annotations[:2]:
        a.font = dict(size=14)
    stem = DIR / "eigengap_bins"
    fig.write_image(f"{stem}.png", scale=2)
    fig.write_image(f"{stem}.svg")
    print(f"wrote {stem}.png / .svg")


def layer_fig():
    layers = sorted({r["layer"] for r in RES})
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.13,
                        subplot_titles=("Eigengap-recovered K vs layer",
                                        "Cluster recovery (ARI@trueK) vs layer"))
    for fam, color, sym in [("weekday", "#4c78a8", "circle"),
                            ("month", "#e45756", "square")]:
        K = row(fam, layers[0])["true_K"]
        rk = [row(fam, L)["spectral_eigengap"] for L in layers]
        ari = [row(fam, L)["ARI_at_trueK"] for L in layers]
        nm = "days (K=7)" if fam == "weekday" else "months (K=12)"
        fig.add_trace(go.Scatter(x=layers, y=rk, mode="markers+lines", name=nm,
                                 marker=dict(size=10, color=color, symbol=sym),
                                 line=dict(color=color)), row=1, col=1)
        fig.add_hline(y=K, line=dict(color=color, dash="dot", width=1.5), row=1, col=1)
        fig.add_trace(go.Scatter(x=layers, y=ari, mode="markers+lines", name=nm,
                                 marker=dict(size=10, color=color, symbol=sym),
                                 line=dict(color=color), showlegend=False), row=1, col=2)
    fig.update_xaxes(title_text="layer", row=1, col=1, dtick=8)
    fig.update_xaxes(title_text="layer", row=1, col=2, dtick=8)
    fig.update_yaxes(title_text="recovered K (eigengap)", row=1, col=1, gridcolor="#eee")
    fig.update_yaxes(title_text="ARI @ true K", row=1, col=2, gridcolor="#eee",
                     range=[0, 1.05])
    fig.update_layout(
        title=dict(text="Layer matters: bins recover cleanly at L16 "
                        "(dotted = true K), smear in deeper layers",
                   x=0.5, xanchor="center", font=dict(size=16)),
        width=1150, height=460, margin=dict(l=70, r=30, t=80, b=55),
        plot_bgcolor="white", legend=dict(x=0.01, y=0.99))
    for a in fig.layout.annotations[:2]:
        a.font = dict(size=14)
    stem = DIR / "layer_robustness"
    fig.write_image(f"{stem}.png", scale=2)
    fig.write_image(f"{stem}.svg")
    print(f"wrote {stem}.png / .svg")


def eigengap_L40_fig():
    """Eigengap at L40 (the viz layer) after removing the top global PC,
    rm_top1 + k_nn=30. Honest: months is a clean K-free recovery; days is a
    marginal 7-vs-8 call."""
    spec = json.load(open(DIR / "bin_count_final_L40_spectra.json"))
    panels = [("weekday_rmtop1_knn30", 7, "Days — layer 40",
               "gap → 7 (marginal: reads 8<br>at smaller neighborhoods)"),
              ("month_rmtop1_knn30", 12, "Months — layer 40",
               "gap → 12 (robust across<br>layers & neighborhoods)")]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=tuple(p[2] for p in panels))
    for col, (key, K, _title, note) in enumerate(panels, start=1):
        ev = np.array(spec[key]); n = K + 4
        colors = [HI if i < K else BASE for i in range(n)]
        fig.add_trace(go.Scatter(
            x=list(range(n)), y=ev[:n], mode="markers+lines",
            marker=dict(size=9, color=colors), line=dict(color="#ccc", width=1),
            showlegend=False, hovertemplate="λ[%{x}]=%{y:.4f}<extra></extra>"),
            row=1, col=col)
        fig.add_vline(x=K - 0.5, line=dict(color=HI, dash="dash", width=2),
                      row=1, col=col)
        fig.add_annotation(row=1, col=col, x=K - 0.5, y=ev[n - 1] * 0.55, text=note,
                           showarrow=True, arrowhead=2, ax=55, ay=-20,
                           font=dict(size=12, color=HI), align="left")
        fig.update_xaxes(title_text="eigenvalue index  i", row=1, col=col,
                         dtick=2, showgrid=False)
        fig.update_yaxes(title_text="Laplacian eigenvalue λ", row=1, col=col,
                         gridcolor="#eee")
    fig.update_layout(
        title=dict(text="Bin count at the L40 visualization layer — eigengap after "
                        "removing the top global PC (no K supplied)",
                   x=0.5, xanchor="center", font=dict(size=15)),
        width=1150, height=460, margin=dict(l=70, r=30, t=80, b=55),
        plot_bgcolor="white")
    for a in fig.layout.annotations[:2]:
        a.font = dict(size=14)
    stem = DIR / "eigengap_bins_L40"
    fig.write_image(f"{stem}.png", scale=2)
    fig.write_image(f"{stem}.svg")
    print(f"wrote {stem}.png / .svg")


def prediction_strength_L40_fig():
    """Prediction strength vs K at L40 (viz layer), cleaned representation.
    Recovered K = largest K with PS >= 0.8. Recovers days=7 AND months=12."""
    cur = json.load(open(DIR / "bin_count_rule_L40_curves.json"))
    THR = 0.8
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=("Days — layer 40", "Months — layer 40"))
    for col, fam in enumerate(["weekday", "month"], start=1):
        ps = cur[fam]["prediction_strength"]; K = cur[fam]["true_K"]
        ks = sorted(int(k) for k in ps); y = [ps[str(k)] for k in ks]
        recovered = max([k for k in ks if ps[str(k)] >= THR])
        colors = [HI if k == recovered else (BASE if ps[str(k)] >= THR else GREY)
                  for k in ks]
        fig.add_trace(go.Bar(x=ks, y=y, marker_color=colors, showlegend=False,
                             hovertemplate="K=%{x}: PS=%{y:.2f}<extra></extra>"),
                      row=1, col=col)
        fig.add_hline(y=THR, line=dict(color="#888", dash="dash", width=1.5),
                      annotation_text="PS = 0.8", annotation_position="top left",
                      row=1, col=col)
        fig.add_annotation(row=1, col=col, x=recovered, y=ps[str(recovered)],
                           text=f"largest stable K → {recovered} bins",
                           showarrow=True, arrowhead=2, ax=60, ay=55,
                           font=dict(size=13, color=HI), align="left")
        fig.update_xaxes(title_text="number of clusters K", row=1, col=col,
                         dtick=2, showgrid=False)
        fig.update_yaxes(title_text="prediction strength", row=1, col=col,
                         range=[0, 1.05], gridcolor="#eee")
    fig.update_layout(
        title=dict(text="Bin count at L40 via prediction strength "
                        "(largest K that clusters stably) — no K supplied",
                   x=0.5, xanchor="center", font=dict(size=15)),
        width=1150, height=460, margin=dict(l=70, r=30, t=80, b=55),
        plot_bgcolor="white")
    for a in fig.layout.annotations[:2]:
        a.font = dict(size=14)
    stem = DIR / "prediction_strength_L40"
    fig.write_image(f"{stem}.png", scale=2)
    fig.write_image(f"{stem}.svg")
    print(f"wrote {stem}.png / .svg")


def simple_rule_layers_fig():
    """Recovered K vs layer for the simplest robust rule: silhouette on
    unit-normalized + top-PC-removed activations. Exact for both concepts
    through L40; degrades only in the deepest layers."""
    res = json.load(open(DIR / "bin_count_simple_results.json"))
    col = "sil_unitnorm_rm_top1"
    layers = sorted({r["layer"] for r in res})
    fig = go.Figure()
    for fam, color, sym, nm, K in [("weekday", "#4c78a8", "circle", "days (K=7)", 7),
                                   ("month", "#e45756", "square", "months (K=12)", 12)]:
        y = [next(r[col] for r in res if r["family"] == fam and r["layer"] == L)
             for L in layers]
        # filled markers where exact, hollow where off
        mc = [color if v == K else "white" for v in y]
        fig.add_hline(y=K, line=dict(color=color, dash="dot", width=1.5))
        fig.add_trace(go.Scatter(
            x=layers, y=y, mode="markers+lines", name=nm,
            line=dict(color=color, width=2),
            marker=dict(size=13, color=mc, symbol=sym,
                        line=dict(color=color, width=2)),
            hovertemplate=f"{nm}<br>L%{{x}}: recovered K=%{{y}}<extra></extra>"))
    fig.add_vrect(x0=14, x1=41, fillcolor="#2ca02c", opacity=0.06, line_width=0,
                  annotation_text="exact for both (incl. L40 viz layer)",
                  annotation_position="top left",
                  annotation_font=dict(size=12, color="#2ca02c"))
    fig.update_xaxes(title_text="layer", dtick=8, showgrid=False)
    fig.update_yaxes(title_text="recovered K  (filled = exact)", gridcolor="#eee",
                     dtick=2)
    fig.update_layout(
        title=dict(text="Simplest robust rule: silhouette on unit-norm + top-PC-"
                        "removed acts — exact K at every layer through L40 (no K supplied)",
                   x=0.5, xanchor="center", font=dict(size=14)),
        width=1050, height=470, margin=dict(l=70, r=30, t=70, b=55),
        plot_bgcolor="white", legend=dict(x=0.01, y=0.99))
    stem = DIR / "simple_rule_layers"
    fig.write_image(f"{stem}.png", scale=2)
    fig.write_image(f"{stem}.svg")
    print(f"wrote {stem}.png / .svg")


if __name__ == "__main__":
    eigengap_fig()
    layer_fig()
    eigengap_L40_fig()
    prediction_strength_L40_fig()
    simple_rule_layers_fig()

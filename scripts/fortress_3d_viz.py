"""3D visualization of FORTRESS antipodal-pair manifold at L40.

Shows the model's behavior-contrastive EA structure: 2099 activations at L40,
colored by aware/non-aware label. Two projections:

  (A) SAVE basis (w_fortress as probe) — should show clear class separation
       along the top SAVE direction.
  (B) Cross-probe basis [w_fortress, w_tvc, w_fs_chord] — visualizes the
       mutual orthogonality of the 3 'EA directions' geometrically.

Points are connected by lines per (prompt_id) pair to show the same prompt's
aware vs non-aware rollouts (since FORTRESS pairs are within-prompt).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from safetensors.numpy import load_file

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))
from load_acts_multi import read_last_token_multi_rollout
from save_recovery import save_recovery
from load_acts import load_labels, read_last_token_per_seq

LAYER = 40
FORTRESS_SHARD = Path("/data/artifacts/rohan/activation-harvester/fortress_balanced_v1/shard_dp00")
FORTRESS_LABELS = Path("/home/rkathuria/santi/data/probe_train/rollout_labels_fortress_balanced.jsonl")
EA_SHARD = Path("/data/artifacts/rohan/manifolds/eval_awareness_olmo31_32b_v1/shard_dp00")
EA_LABELS = Path("/home/rkathuria/manifolds/data/eval_awareness_labels.jsonl")
PROBE_DIR = Path("/home/rkathuria/santi/logs/steering/per_layer_steer_files")
OUT_DIR = Path("/home/rkathuria/manifolds/figures/v9_fortress")


def normalize(v):
    return v / (np.linalg.norm(v) + 1e-12)


def gram_schmidt(vectors):
    """Orthogonalize via modified Gram-Schmidt; returns [d, k] basis."""
    out = []
    for v in vectors:
        u = v.copy()
        for w in out:
            u = u - (u @ w) * w
        n = np.linalg.norm(u)
        if n > 1e-9:
            out.append(u / n)
    return np.stack(out, axis=1) if out else np.zeros((vectors[0].shape[0], 0))


def make_3d(coords_aware, coords_non, w_chord_3d, w_other1_3d, w_other2_3d,
              pair_lines, title, axis_labels):
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=coords_non[:, 0], y=coords_non[:, 1], z=coords_non[:, 2],
        mode="markers",
        marker=dict(size=3, color="#4477aa", opacity=0.55, symbol="circle"),
        name=f"non-aware ({len(coords_non)})",
        text=[f"non-aware"] * len(coords_non), hovertemplate="%{text}",
    ))
    fig.add_trace(go.Scatter3d(
        x=coords_aware[:, 0], y=coords_aware[:, 1], z=coords_aware[:, 2],
        mode="markers",
        marker=dict(size=3, color="#cc3311", opacity=0.55, symbol="diamond"),
        name=f"aware ({len(coords_aware)})",
        text=[f"aware"] * len(coords_aware), hovertemplate="%{text}",
    ))

    # Draw a sample of antipodal-pair lines (don't draw all, too cluttered)
    if pair_lines:
        sample = pair_lines[: min(80, len(pair_lines))]
        xs, ys, zs = [], [], []
        for (p_a, p_n) in sample:
            xs.extend([p_a[0], p_n[0], None])
            ys.extend([p_a[1], p_n[1], None])
            zs.extend([p_a[2], p_n[2], None])
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(color="rgba(120, 80, 30, 0.18)", width=1.2),
            name=f"sample pairs (n={len(sample)})", hoverinfo="skip",
        ))

    # Draw the centroids and chord direction
    c_aware = coords_aware.mean(0); c_non = coords_non.mean(0)
    fig.add_trace(go.Scatter3d(
        x=[c_non[0], c_aware[0]], y=[c_non[1], c_aware[1]], z=[c_non[2], c_aware[2]],
        mode="markers+lines+text",
        marker=dict(size=10, color=["#1144aa", "#aa1100"], symbol=["square", "square"],
                     line=dict(color="black", width=2)),
        line=dict(color="black", width=4),
        text=["non-aware mean", "aware mean"], textposition="top center",
        name="centroids + chord",
    ))

    # Project the named directions as arrows from origin
    arrow_scale = max(np.linalg.norm(c_aware - c_non) * 1.2, 1.0)
    for name, vec, color in [("FORTRESS dir (top SAVE)", w_chord_3d, "#cc3311"),
                              ("testing_vs_conv proj", w_other1_3d, "#117755"),
                              ("FS_chord proj", w_other2_3d, "#cc66cc")]:
        if np.linalg.norm(vec) < 1e-9:
            continue
        v = vec / np.linalg.norm(vec) * arrow_scale
        origin = (c_aware + c_non) / 2
        fig.add_trace(go.Scatter3d(
            x=[origin[0], origin[0] + v[0]],
            y=[origin[1], origin[1] + v[1]],
            z=[origin[2], origin[2] + v[2]],
            mode="lines+text", line=dict(color=color, width=5),
            text=["", name], textposition="middle right",
            name=name,
        ))

    fig.update_layout(
        title=title, scene=dict(
            xaxis_title=axis_labels[0],
            yaxis_title=axis_labels[1],
            zaxis_title=axis_labels[2],
            aspectmode="data",
        ),
        width=1000, height=780,
        legend=dict(x=0.02, y=0.95),
    )
    return fig


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("loading FORTRESS L40 multi-rollout...")
    X_f, keys = read_last_token_multi_rollout(FORTRESS_SHARD, LAYER, position_from_end=0)
    aware_map = {}
    with open(FORTRESS_LABELS) as f:
        for line in f:
            d = json.loads(line)
            aware_map[(int(d["prompt_id"]), int(d["completion_idx"]))] = bool(d["aware"])
    matched = [(i, keys[i], aware_map[keys[i]]) for i in range(len(keys)) if keys[i] in aware_map]
    idx = np.array([m[0] for m in matched])
    pair_keys = [m[1] for m in matched]
    aware = np.array([m[2] for m in matched])
    Xa = X_f[idx].astype(np.float64)
    print(f"  matched {Xa.shape[0]} acts, aware={aware.sum()} non={(~aware).sum()}")

    mean_aware = Xa[aware].mean(0); mean_non = Xa[~aware].mean(0)
    w_fortress = normalize(mean_aware - mean_non)
    print(f"  FORTRESS chord ‖={np.linalg.norm(mean_aware - mean_non):.2f}")

    w_tvc = normalize(load_file(str(PROBE_DIR / f"testing_vs_conversation_opus46_L{LAYER}.safetensors"))[f"layer_{LAYER}"].astype(np.float64))
    w_hua = normalize(load_file(str(PROBE_DIR / f"hua_et_al_L{LAYER}.safetensors"))[f"layer_{LAYER}"].astype(np.float64))

    print("loading EA L40 for FS chord direction...")
    X_ea, pids = read_last_token_per_seq(EA_SHARD, LAYER, position_from_end=10)
    ea_labels = load_labels(EA_LABELS)
    fs = np.array([ea_labels[p]["frame_score"] for p in pids])
    cents_ea = np.stack([X_ea[fs == k].astype(np.float64).mean(0) for k in range(7)])
    w_fs = normalize(cents_ea[6] - cents_ea[0])

    # ===== Projection (A): SAVE basis with FORTRESS probe =====
    print("\n=== running SAVE on FORTRESS with w_fortress ===")
    save_res = save_recovery(Xa, w_fortress, n_slices=10, d_M=3)
    V_save = np.column_stack([w_fortress, save_res.save_dirs[:, 0], save_res.save_dirs[:, 1]])
    V_save = gram_schmidt([V_save[:, i] for i in range(V_save.shape[1])])  # orthonormalize
    print(f"  SAVE eigval[0]/eigval[20] = {save_res.eigvals[0] / max(save_res.eigvals[20], 1e-12):.1f}")

    Z_save = Xa @ V_save                       # [N, 3]
    coords_aware = Z_save[aware]; coords_non = Z_save[~aware]

    # Build pair lines for sample of within-prompt antipodal pairs
    pair_index = {}
    for i, (pid, cidx) in enumerate(pair_keys):
        pair_index.setdefault(pid, {"aware": [], "non": []})
        ("aware" if aware[i] else "non")
        if aware[i]:
            pair_index[pid]["aware"].append(i)
        else:
            pair_index[pid]["non"].append(i)
    sample_pairs = []
    for pid, d in pair_index.items():
        if d["aware"] and d["non"]:
            sample_pairs.append((Z_save[d["aware"][0]], Z_save[d["non"][0]]))

    fig_save = make_3d(coords_aware, coords_non,
                        w_chord_3d=np.array([1.0, 0.0, 0.0]),  # w_fortress is V_save column 0
                        w_other1_3d=V_save.T @ w_tvc,           # project tvc onto SAVE basis
                        w_other2_3d=V_save.T @ w_fs,
                        pair_lines=sample_pairs,
                        title=f"FORTRESS L{LAYER} — SAVE basis (probe = FORTRESS aware-non-aware), eigval ratio = {save_res.eigvals[0] / max(save_res.eigvals[20], 1e-12):.0f}×",
                        axis_labels=["w_fortress (probe)", "SAVE dir 1", "SAVE dir 2"])

    # ===== Projection (B): cross-probe basis [w_fortress, w_tvc, w_fs] =====
    V_cross = gram_schmidt([w_fortress, w_tvc, w_fs])
    Z_cross = Xa @ V_cross
    coords_aware_c = Z_cross[aware]; coords_non_c = Z_cross[~aware]
    sample_pairs_c = []
    for pid, d in pair_index.items():
        if d["aware"] and d["non"]:
            sample_pairs_c.append((Z_cross[d["aware"][0]], Z_cross[d["non"][0]]))

    fig_cross = make_3d(coords_aware_c, coords_non_c,
                         w_chord_3d=V_cross.T @ w_fortress,
                         w_other1_3d=V_cross.T @ w_tvc,
                         w_other2_3d=V_cross.T @ w_fs,
                         pair_lines=sample_pairs_c,
                         title=f"FORTRESS L{LAYER} — cross-probe basis (3 EA directions, mutually orthogonalized via GS)",
                         axis_labels=["w_fortress", "w_testing_vs_conv ⊥", "w_FS_chord ⊥"])

    fig_save.write_html(OUT_DIR / "fortress_3d_save.html", include_plotlyjs="cdn")
    fig_cross.write_html(OUT_DIR / "fortress_3d_crossbasis.html", include_plotlyjs="cdn")

    # Combined viewer
    save_html = fig_save.to_html(include_plotlyjs="cdn", full_html=False)
    cross_html = fig_cross.to_html(include_plotlyjs=False, full_html=False)
    combined = f"""<!doctype html><html><head>
<title>FORTRESS 3D — antipodal-pair EA manifold L{LAYER}</title>
<style>body {{ font-family: sans-serif; margin: 20px; }} h2 {{ margin-top: 30px; }}
.panel {{ font-size: 13px; color: #555; max-width: 900px; }}</style>
</head><body>
<h1>FORTRESS antipodal-pair EA manifold — OLMo-3.1-Think-32B, L{LAYER}</h1>
<p class='panel'>2099 activations at L{LAYER}, last-token of CoT rollouts. Each
prompt has 2 rollouts: one labeled aware, one labeled non-aware (judged by Hua
classifier). Same prompt → same surface text → contrast is pure model behavior.</p>

<h2>Projection A: SAVE basis (w_fortress as probe)</h2>
<p class='panel'>Top SAVE eigval is {save_res.eigvals[0] / max(save_res.eigvals[20], 1e-12):.0f}×
above noise floor — clean 1-D structure. The w_fortress axis (x) should show
maximal aware/non-aware separation; w_tvc and w_fs_chord arrows show their
projection onto this basis (mostly zero if orthogonal, as we measured).</p>
{save_html}

<h2>Projection B: cross-probe basis (3 EA directions)</h2>
<p class='panel'>Three different "EA" directions, Gram-Schmidt orthogonalized:
w_fortress (model behavior contrast), w_testing_vs_conv (Apollo probe), w_FS_chord
(our hand-labeled framing). Aware/non-aware separation should appear ONLY along
w_fortress — the other two axes should show no class structure since they're
orthogonal to FORTRESS by measurement (cos ≈ ±0.005).</p>
{cross_html}
</body></html>"""
    (OUT_DIR / "fortress_3d_combined.html").write_text(combined)
    print(f"\nwrote {OUT_DIR}/fortress_3d_combined.html")
    print(f"wrote {OUT_DIR}/fortress_3d_save.html")
    print(f"wrote {OUT_DIR}/fortress_3d_crossbasis.html")


if __name__ == "__main__":
    main()

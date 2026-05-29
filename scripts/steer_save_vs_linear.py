"""Manifold-steering experiment using the SAVE-recovered manifold.

Replicates the paper's linear-vs-manifold steering comparison:
  - K=50 waypoints from Mon centroid to Fri centroid
  - Linear path: straight line in ambient activation space
  - Manifold path: walk along the cycle Mon→Tue→Wed→Thu→Fri (passing through
    intermediate concept centroids on the SAVE-recovered manifold)
  - At each waypoint: replace the residual at the steering layer with the
    waypoint activation, continue forward, read next-token probabilities
    over the 7 day tokens, normalize.

Outputs a 2-row heatmap: row 1 = linear path day-probs over time,
row 2 = manifold path day-probs over time.
Manifold steering should show smooth Mon→...→Fri flow; linear should
"teleport" — probability mass jumps directly Mon→Fri without intermediates.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))
from load_acts import load_labels, read_last_token_per_seq
from save_recovery import save_recovery


WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
LLAMA_PATH = ("/data/artifacts/rohan/santi/hf_cache/hub/"
              "models--meta-llama--Llama-3.1-8B/snapshots/"
              "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b")
SHARD = Path("/data/artifacts/rohan/manifolds/paper_llama31_8b_base_v4/shard_dp00")
LABELS = Path("/home/rkathuria/manifolds/data/labels_llama_paper.jsonl")


def chord_probe(X, vals):
    arr = np.array(vals, np.float64)
    lo, hi = np.quantile(arr, [0.2, 0.8])
    w = X[arr >= hi].mean(0) - X[arr <= lo].mean(0)
    return w / (np.linalg.norm(w) + 1e-12)


def build_manifold_path(centroids_ambient, source_idx, target_idx, n_waypoints=50, direction="forward"):
    """Walk Mon→Tue→Wed→...→Fri (4 hops) through concept centroids — piecewise linear."""
    C = len(centroids_ambient)
    if direction == "forward":
        path_idx = [(source_idx + k) % C for k in range(target_idx - source_idx + 1)]
    else:
        path_idx = [(source_idx - k) % C for k in range(source_idx - target_idx + 1)]
    path_centroids = np.stack([centroids_ambient[i] for i in path_idx])

    seg_lens = np.linalg.norm(np.diff(path_centroids, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    L = cum[-1]
    t_eval = np.linspace(0, L, n_waypoints)
    waypoints = np.zeros((n_waypoints, path_centroids.shape[1]))
    for j, t in enumerate(t_eval):
        seg = int(np.searchsorted(cum, t, side="right") - 1)
        seg = min(seg, len(path_centroids) - 2)
        local = (t - cum[seg]) / max(seg_lens[seg], 1e-12)
        waypoints[j] = (1 - local) * path_centroids[seg] + local * path_centroids[seg + 1]
    return waypoints, path_idx


def build_paper_spline_path(centroids_ambient, source_idx, target_idx, n_waypoints=50,
                              direction="forward"):
    """Paper's manifold steering — periodic cubic spline through ALL 7 concept
    centroids in calendar order, then sample 50 points along the source→target arc."""
    from scipy.interpolate import CubicSpline
    C = len(centroids_ambient)
    # Periodic cubic spline through Mon, Tue, …, Sun, Mon
    t_pts = np.arange(C + 1)
    Y = np.vstack([centroids_ambient, centroids_ambient[:1]])
    spline = CubicSpline(t_pts, Y, bc_type="periodic", axis=0)

    if direction == "forward":
        t_start, t_end = float(source_idx), float(target_idx)
        if t_end < t_start:
            t_end += C
    else:
        t_start, t_end = float(source_idx), float(target_idx)
        if t_end > t_start:
            t_end -= C
    t_eval = np.linspace(t_start, t_end, n_waypoints) % C
    return spline(t_eval), [(source_idx + k) % C for k in range(int(t_end - t_start) + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steer-layer", type=int, default=19,
                    help="layer at which to inject the activation (paper uses L28; "
                         "we use the layer where SAVE was best = L19)")
    ap.add_argument("--n-waypoints", type=int, default=50)
    ap.add_argument("--source", default="Monday")
    ap.add_argument("--target", default="Friday")
    ap.add_argument("--out", default="/home/rkathuria/manifolds/figures/steer_save_vs_linear.png")
    args = ap.parse_args()

    # ---- 1. Compute SAVE concept centroids in ambient at the steering layer ----
    labels = load_labels(LABELS)
    X_full, pids = read_last_token_per_seq(SHARD, args.steer_layer)
    keep = [i for i, p in enumerate(pids) if labels[p].get("family") == "weekday"]
    X = X_full[keep]
    vals = [WEEKDAY_ORDER.index(labels[pids[i]]["concept"]) for i in keep]
    print(f"L{args.steer_layer} weekday: n={X.shape[0]}, d={X.shape[1]}")

    # Per-concept centroids in ambient activation space
    cents_ambient = np.stack([X[np.array(vals) == k].mean(0) for k in range(7)])  # [7, d]

    # ---- 2. Build linear and manifold paths ----
    src = WEEKDAY_ORDER.index(args.source); tgt = WEEKDAY_ORDER.index(args.target)
    print(f"steering {args.source}({src}) → {args.target}({tgt})")

    linear_path = np.array([
        (1 - t) * cents_ambient[src] + t * cents_ambient[tgt]
        for t in np.linspace(0, 1, args.n_waypoints)
    ])
    manifold_path, mf_idx = build_manifold_path(
        cents_ambient, src, tgt, args.n_waypoints, direction="forward")
    paper_path, _ = build_paper_spline_path(
        cents_ambient, src, tgt, args.n_waypoints, direction="forward")
    print(f"manifold path passes through: {[WEEKDAY_ORDER[i] for i in mf_idx]}")

    # ---- 3. Load Llama and set up steering hook ----
    print("loading Llama-3.1-8B base...")
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_PATH, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    day_token_ids = {d: tokenizer.encode(" " + d, add_special_tokens=False)[0]
                     for d in WEEKDAY_ORDER}
    print(f"day token ids: {day_token_ids}")

    intercept = {"vec": None}
    def hook(_, __, output):
        if intercept["vec"] is None:
            return output
        if isinstance(output, tuple):
            h, *rest = output
            h = h.clone()
            h[:, -1, :] = intercept["vec"].to(h.dtype).to(h.device)
            return (h, *rest)
        h = output.clone()
        h[:, -1, :] = intercept["vec"].to(h.dtype).to(h.device)
        return h

    handle = model.model.layers[args.steer_layer].register_forward_hook(hook)

    # ---- 4. Score each waypoint: average over multiple base prompts ----
    base_prompts = [
        "It's 5pm on day",
        "It's 9am on day",
        "It's noon on day",
        "It's midnight on day",
        "It's a sunny afternoon on day",
    ]

    def get_day_probs(path):
        """Return [K, 7] day probabilities along the path."""
        out = np.zeros((len(path), 7))
        for k, wp in enumerate(path):
            wp_t = torch.from_numpy(wp).to(torch.bfloat16).cuda()
            intercept["vec"] = wp_t
            avg = np.zeros(7)
            for p in base_prompts:
                ids = tokenizer(p, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    logits = model(**ids).logits[:, -1]
                sm = torch.softmax(logits.float(), dim=-1)[0].cpu().numpy()
                # Aggregate over day tokens, normalize
                day_p = np.array([sm[day_token_ids[d]] for d in WEEKDAY_ORDER])
                day_p /= day_p.sum() + 1e-12  # only-among-days simplex
                avg += day_p
            avg /= len(base_prompts)
            out[k] = avg
            if k % 10 == 0:
                top = WEEKDAY_ORDER[int(np.argmax(avg))]
                print(f"    waypoint {k:>3}/{len(path)}: dominant = {top} ({avg.max():.2f})")
        return out

    print("\nrunning LINEAR steering...")
    linear_probs = get_day_probs(linear_path)
    print("\nrunning OURS — SAVE manifold (piecewise-linear cycle arc)...")
    manifold_probs = get_day_probs(manifold_path)
    print("\nrunning PAPER — periodic cubic spline through concept centroids...")
    paper_probs = get_day_probs(paper_path)

    handle.remove()

    # ---- 5. Plot — three rows: linear / paper-spline / ours ----
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for ax, probs, name in zip(axes,
                                [linear_probs, paper_probs, manifold_probs],
                                ["LINEAR (chord)",
                                 "PAPER manifold (periodic cubic spline through 7 centroids)",
                                 "OURS — SAVE manifold (piecewise-linear cycle arc)"]):
        im = ax.imshow(probs.T, aspect="auto", cmap="viridis", vmin=0, vmax=1,
                       extent=[0, args.n_waypoints, 6.5, -0.5])
        ax.set_yticks(range(7))
        ax.set_yticklabels(WEEKDAY_ORDER)
        ax.set_ylabel("day token prob")
        ax.set_title(f"{name} — {args.source} → {args.target}")
    axes[2].set_xlabel("waypoint along path")
    fig.colorbar(im, ax=axes, label="prob (over day-tokens, normalized)")
    fig.suptitle(f"Steering comparison: linear vs paper-style manifold vs SAVE-manifold\n"
                 f"Llama-3.1-8B base, layer {args.steer_layer}")
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"\nwrote {args.out}")

    # Compute a simple "smoothness" / probability-flow metric:
    # mean per-waypoint Total-Variation distance from previous waypoint.
    def tv_smoothness(probs):
        return float(np.abs(np.diff(probs, axis=0)).sum(axis=1).mean())
    print(f"\nTV smoothness (lower = smoother flow):")
    print(f"  linear:        {tv_smoothness(linear_probs):.4f}")
    print(f"  paper spline:  {tv_smoothness(paper_probs):.4f}")
    print(f"  ours (SAVE):   {tv_smoothness(manifold_probs):.4f}")

    # "Coverage of intermediate concepts" — how often does Tue/Wed/Thu dominate?
    def intermediate_coverage(probs):
        dom = np.argmax(probs, axis=1)
        # Tuesday=1, Wednesday=2, Thursday=3
        return float(np.mean((dom >= 1) & (dom <= 3)))
    print(f"\nFraction of waypoints where Tue/Wed/Thu dominates (probe of intermediate-concept flow):")
    print(f"  linear:        {intermediate_coverage(linear_probs):.3f}")
    print(f"  paper spline:  {intermediate_coverage(paper_probs):.3f}")
    print(f"  ours (SAVE):   {intermediate_coverage(manifold_probs):.3f}")

    np.savez(str(args.out).replace(".png", ".npz"),
             linear_probs=linear_probs, manifold_probs=manifold_probs,
             paper_probs=paper_probs,
             linear_path=linear_path, manifold_path=manifold_path,
             paper_path=paper_path)
    print(f"wrote {str(args.out).replace('.png', '.npz')}")


if __name__ == "__main__":
    main()

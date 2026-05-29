"""Days/cycle steering on OLMo-32B via ALL-POSITION ADDITIVE steering.

Sanity check for the all-position additive setup we're using on EA.
If days still produces clean Mon→Tue→Wed→Thu→Fri with this method, we
know the methodology is valid; if not, something else is wrong.

For each waypoint k along path:
  offset = path[k] - path[0]
  add offset at EVERY token position at the steering layer
  read next-token day-name probabilities
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import CubicSpline
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))
from load_acts import load_labels, read_last_token_per_seq

OLMO_PATH = ("/data/artifacts/rohan/santi/hf_cache/hub/"
             "models--allenai--Olmo-3.1-32B-Think/snapshots")
WEEKDAY_SHARD = Path("/data/artifacts/rohan/manifolds/concepts_olmo31_32b_v3/shard_dp00")
WEEKDAY_LABELS = Path("/home/rkathuria/manifolds/data/labels_v3.jsonl")
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def linear_path(c_src, c_tgt, K):
    return np.array([(1 - t) * c_src + t * c_tgt for t in np.linspace(0, 1, K)])


def cycle_arc_path(centroids, src, tgt, K, cyclic=True):
    C = len(centroids)
    if cyclic:
        path_idx = [(src + k) % C for k in range(((tgt - src) % C) + 1)]
    else:
        path_idx = list(range(src, tgt + 1)) if tgt >= src else list(range(src, tgt - 1, -1))
    pts = np.stack([centroids[i] for i in path_idx])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    L = cum[-1]
    out = np.zeros((K, pts.shape[1]))
    for j, t in enumerate(np.linspace(0, L, K)):
        s = int(np.searchsorted(cum, t, side="right") - 1)
        s = min(s, len(pts) - 2)
        local = (t - cum[s]) / max(seg[s], 1e-12)
        out[j] = (1 - local) * pts[s] + local * pts[s + 1]
    return out


def paper_spline_path(centroids, src, tgt, K, cyclic=True):
    C = len(centroids)
    if cyclic:
        t_pts = np.arange(C + 1)
        Y = np.vstack([centroids, centroids[:1]])
        sp = CubicSpline(t_pts, Y, bc_type="periodic", axis=0)
        ts, te = float(src), float(tgt)
        if te <= ts: te += C
        return sp(np.linspace(ts, te, K) % C)
    else:
        sp = CubicSpline(np.arange(C), centroids, bc_type="natural", axis=0)
        return sp(np.linspace(src, tgt, K))


def find_olmo_snapshot():
    return sorted(Path(OLMO_PATH).iterdir())[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steer-layer", type=int, default=40)
    ap.add_argument("--n-waypoints", type=int, default=21)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--source", default="Monday")
    ap.add_argument("--target", default="Friday")
    ap.add_argument("--out-dir", default="/home/rkathuria/manifolds/figures/days_olmo_allpos")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    labels = load_labels(WEEKDAY_LABELS)
    K = args.n_waypoints

    X, pids = read_last_token_per_seq(WEEKDAY_SHARD, args.steer_layer)
    keep = [i for i, p in enumerate(pids) if labels[p].get("family") == "weekday"]
    Xw = X[keep]
    cidx = np.array([WEEKDAY_ORDER.index(labels[pids[i]]["concept"]) for i in keep])
    cents = np.stack([Xw[cidx == k].mean(0) for k in range(7)])

    src = WEEKDAY_ORDER.index(args.source); tgt = WEEKDAY_ORDER.index(args.target)
    paths = {
        "linear": linear_path(cents[src], cents[tgt], K),
        "paper":  paper_spline_path(cents, src, tgt, K, cyclic=True),
        "ours":   cycle_arc_path(cents, src, tgt, K, cyclic=True),
    }

    snap = find_olmo_snapshot()
    print(f"loading OLMo-32B...")
    tokenizer = AutoTokenizer.from_pretrained(str(snap))
    model = AutoModelForCausalLM.from_pretrained(
        str(snap), dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    # All-position additive hook
    hook_state = {"offset": None, "alpha": args.alpha}
    def steer_hook(_, __, output):
        if hook_state["offset"] is None:
            return output
        delta = (hook_state["offset"].to(torch.bfloat16).cuda()
                  * float(hook_state["alpha"]))
        if isinstance(output, tuple):
            h, *rest = output
            h = h.clone() + delta[None, None, :]
            return (h, *rest)
        return output.clone() + delta[None, None, :]
    handle = model.model.layers[args.steer_layer].register_forward_hook(steer_hook)

    test_prompts_user = [
        "It's 5pm on", "It's noon on", "It's midnight on",
        "It's a sunny afternoon on", "It's the late shift on",
    ]
    test_chat = [tokenizer.apply_chat_template(
        [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        for p in test_prompts_user]
    day_token_ids = [tokenizer.encode(" " + d, add_special_tokens=False)[0] for d in WEEKDAY_ORDER]

    # Baseline
    print("\n--- baseline (no steering) ---")
    hook_state["offset"] = None
    base_probs = []
    for ct in test_chat:
        ids = tokenizer(ct, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.no_grad():
            logits = model(**ids).logits[:, -1]
        sm = torch.softmax(logits.float(), dim=-1)[0]
        days = np.array([sm[t].item() for t in day_token_ids])
        days /= days.sum() + 1e-12
        base_probs.append(days)
        print(f"  dominant={WEEKDAY_ORDER[int(np.argmax(days))]} ({days.max():.2f})")
    base_probs = np.array(base_probs)

    results = {}
    for method, path in paths.items():
        print(f"\n--- {method.upper()} (all-position additive) ---")
        all_probs = np.zeros((K, len(test_chat), 7))
        for k, wp in enumerate(path):
            offset = wp - path[0]
            hook_state["offset"] = torch.from_numpy(offset).to(torch.bfloat16).cuda()
            for ti, ct in enumerate(test_chat):
                ids = tokenizer(ct, return_tensors="pt", add_special_tokens=False).to("cuda")
                with torch.no_grad():
                    logits = model(**ids).logits[:, -1]
                sm = torch.softmax(logits.float(), dim=-1)[0]
                days = np.array([sm[t].item() for t in day_token_ids])
                days /= days.sum() + 1e-12
                all_probs[k, ti] = days
            if k in (0, K // 2, K - 1):
                avg = all_probs[k].mean(0)
                print(f"  wp {k:>2}/{K-1}: dominant={WEEKDAY_ORDER[int(np.argmax(avg))]} ({avg.max():.2f})")
        results[method] = all_probs

    handle.remove()

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for ax, method in zip(axes, ["linear", "paper", "ours"]):
        probs = results[method].mean(axis=1)
        im = ax.imshow(probs.T, aspect="auto", cmap="viridis",
                        vmin=0, vmax=max(0.5, probs.max()),
                        extent=[0, K, 6.5, -0.5])
        ax.set_yticks(range(7)); ax.set_yticklabels(WEEKDAY_ORDER, fontsize=8)
        ax.set_ylabel("day token prob")
        ax.set_title(f"{method.upper()} all-pos additive — OLMo days {args.source}→{args.target}, L{args.steer_layer}",
                      fontsize=11)
    axes[2].set_xlabel("waypoint")
    fig.colorbar(im, ax=axes)
    fig.suptitle(f"Days steering ALL-POSITION ADDITIVE  alpha={args.alpha}", fontsize=12)
    fig.savefig(out_dir / f"days_allpos_{args.source}_to_{args.target}.png",
                 dpi=140, bbox_inches="tight")
    plt.close(fig)

    intermediates = set((src + k) % 7 for k in range(1, ((tgt - src) % 7)))
    print(f"\nintermediates = {[WEEKDAY_ORDER[i] for i in intermediates]}")
    for method in ["linear", "paper", "ours"]:
        probs = results[method].mean(axis=1)
        dom = np.argmax(probs, axis=1)
        cov = float(np.mean([d in intermediates for d in dom]))
        print(f"  {method}: intermediate-coverage = {cov:.3f}")

    np.savez(out_dir / "days_allpos_data.npz",
              **{m: results[m] for m in ["linear", "paper", "ours"]},
              base_probs=base_probs)
    print(f"\nwrote {out_dir}/")


if __name__ == "__main__":
    main()

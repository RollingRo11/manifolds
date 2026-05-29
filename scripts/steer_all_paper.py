"""Full battery of steering experiments — replicate the paper's Linear vs
Manifold steering across all manifold families we have data for.

For each (family, source_value, target_value) tuple:
  - Linear  : straight line in ambient between source and target centroids
  - Paper   : periodic cubic spline through ALL family centroids in calendar
              order, walking from source to target
  - Ours    : piecewise-linear cycle arc through SAVE-recovered manifold
              (collapses to "all family centroids in order" for cyclic data)

Reads next-token logits over the family's value tokens at each waypoint and
plots probability flow.
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
from scipy.interpolate import CubicSpline
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))
from load_acts import load_labels, read_last_token_per_seq

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

LLAMA_PATH = ("/data/artifacts/rohan/santi/hf_cache/hub/"
              "models--meta-llama--Llama-3.1-8B/snapshots/"
              "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b")
SHARD = Path("/data/artifacts/rohan/manifolds/paper_llama31_8b_base_v4/shard_dp00")
LABELS = Path("/home/rkathuria/manifolds/data/labels_llama_paper.jsonl")


# ----- Path constructors ---------------------------------------------------- #


def linear_path(c_src, c_tgt, K):
    return np.array([(1 - t) * c_src + t * c_tgt for t in np.linspace(0, 1, K)])


def cycle_arc_path(centroids, src_idx, tgt_idx, K, cyclic=True):
    """Piecewise-linear walk through centroids[src..tgt] (forward in calendar)."""
    C = len(centroids)
    if cyclic:
        path_idx = [(src_idx + k) % C for k in range(((tgt_idx - src_idx) % C) + 1)]
    else:
        if tgt_idx >= src_idx:
            path_idx = list(range(src_idx, tgt_idx + 1))
        else:
            path_idx = list(range(src_idx, tgt_idx - 1, -1))
    pts = np.stack([centroids[i] for i in path_idx])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    L = cum[-1]
    t_eval = np.linspace(0, L, K)
    out = np.zeros((K, pts.shape[1]))
    for j, t in enumerate(t_eval):
        s = int(np.searchsorted(cum, t, side="right") - 1)
        s = min(s, len(pts) - 2)
        local = (t - cum[s]) / max(seg[s], 1e-12)
        out[j] = (1 - local) * pts[s] + local * pts[s + 1]
    return out, path_idx


def paper_spline_path(centroids, src_idx, tgt_idx, K, cyclic=True):
    """Spline through ALL centroids; sample K points on src→tgt arc."""
    C = len(centroids)
    if cyclic:
        t_pts = np.arange(C + 1)
        Y = np.vstack([centroids, centroids[:1]])
        spline = CubicSpline(t_pts, Y, bc_type="periodic", axis=0)
        t_start = float(src_idx); t_end = float(tgt_idx)
        if t_end <= t_start:
            t_end += C
        t_eval = np.linspace(t_start, t_end, K) % C
        return spline(t_eval)
    else:
        # Open spline (line/helix manifolds — natural BC, no wrap)
        t_pts = np.arange(C)
        spline = CubicSpline(t_pts, centroids, bc_type="natural", axis=0)
        t_eval = np.linspace(src_idx, tgt_idx, K)
        return spline(t_eval)


# ----- Steering experiment ------------------------------------------------- #


def display_labels_for(labels_v):
    return labels_v


def get_path_probs(model, tokenizer, hook_state, target_layer, path,
                    base_prompts, value_token_ids):
    """Run forward at each waypoint, return [K, V] probs over value tokens."""
    K = len(path)
    V = len(value_token_ids)
    out = np.zeros((K, V))
    for k, wp in enumerate(path):
        wp_t = torch.from_numpy(wp).to(torch.bfloat16).cuda()
        hook_state["vec"] = wp_t
        avg = np.zeros(V)
        for p in base_prompts:
            ids = tokenizer(p, return_tensors="pt").to("cuda")
            with torch.no_grad():
                logits = model(**ids).logits[:, -1]
            sm = torch.softmax(logits.float(), dim=-1)[0].cpu().numpy()
            v = np.array([sm[t] for t in value_token_ids])
            v /= v.sum() + 1e-12
            avg += v
        out[k] = avg / len(base_prompts)
    return out


# ----- Per-family configuration ------------------------------------------- #


def family_config(family: str, layer: int, labels: dict, pids: list, X: np.ndarray, tokenizer):
    """Return (centroids, value_labels, value_token_ids, default_pairs, base_prompts, cyclic)."""
    if family == "weekday":
        order = WEEKDAY_ORDER
        keep = [i for i, p in enumerate(pids) if labels[p].get("family") == "weekday"]
        Xf = X[keep]
        cidx = [order.index(labels[pids[i]]["concept"]) for i in keep]
        cents = np.stack([Xf[np.array(cidx) == k].mean(0) for k in range(len(order))])
        toks = [tokenizer.encode(" " + d, add_special_tokens=False)[0] for d in order]
        return cents, order, toks, [
            ("Monday", "Friday"),
            ("Tuesday", "Saturday"),
            ("Friday", "Monday"),  # reverse
        ], [
            "It's 5pm on day", "It's noon on day", "It's midnight on day",
            "It's a sunny afternoon on day", "It's the late shift on day",
        ], True

    if family == "temperature":
        keep = [i for i, p in enumerate(pids) if labels[p].get("family") == "temperature"]
        Xf = X[keep]
        vals = [int(labels[pids[i]]["value"]) for i in keep]
        uniq = sorted(set(vals))
        cents, label_strs, toks = [], [], []
        for v in uniq:
            mask = np.array(vals) == v
            if mask.sum() == 0: continue
            tok_ids = tokenizer.encode(" " + str(v), add_special_tokens=False)
            if not tok_ids: continue
            cents.append(Xf[mask].mean(0))
            label_strs.append(str(v))
            toks.append(tok_ids[0])  # use FIRST token for multi-token values
        cents = np.stack(cents)
        return cents, label_strs, toks, [
            (label_strs[0], label_strs[-1]),
            (label_strs[len(label_strs)//4], label_strs[3*len(label_strs)//4]),
        ], [
            "Today it's", "It's", "The temperature is",
        ], False

    if family == "age":
        keep = [i for i, p in enumerate(pids) if labels[p].get("family") == "age"]
        Xf = X[keep]
        vals = [int(labels[pids[i]]["value"]) for i in keep]
        uniq = sorted(set(vals))
        cents, label_strs, toks = [], [], []
        for v in uniq:
            mask = np.array(vals) == v
            if mask.sum() == 0: continue
            tok_ids = tokenizer.encode(" " + str(v), add_special_tokens=False)
            if not tok_ids: continue
            cents.append(Xf[mask].mean(0))
            label_strs.append(str(v))
            toks.append(tok_ids[0])  # use FIRST token; multi-token vals get partial credit
        cents = np.stack(cents)
        return cents, label_strs, toks, [
            (label_strs[0], label_strs[-1]),
            (label_strs[len(label_strs)//4], label_strs[3*len(label_strs)//4]),
        ], [
            "They are", "He is",
        ], False

    if family == "year":
        keep = [i for i, p in enumerate(pids) if labels[p].get("family") == "year"]
        Xf = X[keep]
        vals = [int(labels[pids[i]]["value"]) for i in keep]
        uniq = sorted(set(vals))
        cents, label_strs, toks = [], [], []
        for v in uniq:
            mask = np.array(vals) == v
            if mask.sum() == 0: continue
            tok_ids = tokenizer.encode(" " + str(v), add_special_tokens=False)
            if not tok_ids: continue
            cents.append(Xf[mask].mean(0))
            label_strs.append(str(v))
            toks.append(tok_ids[0])  # use FIRST token; multi-token vals get partial credit
        cents = np.stack(cents)
        return cents, label_strs, toks, [
            (label_strs[0], label_strs[-1]),
            (label_strs[len(label_strs)//4], label_strs[3*len(label_strs)//4]),
        ], [
            "The date is year",
        ], False

    raise ValueError(f"unknown family {family}")


# ----- Main ---------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steer-layer", type=int, default=19)
    ap.add_argument("--n-waypoints", type=int, default=50)
    ap.add_argument("--out-dir", default="/home/rkathuria/manifolds/figures/steer_battery")
    ap.add_argument("--families", default="weekday,temperature,age,year")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    labels = load_labels(LABELS)
    X_full, pids = read_last_token_per_seq(SHARD, args.steer_layer)

    print("loading Llama-3.1-8B base...")
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_PATH, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    hook_state = {"vec": None}
    def hook(_, __, output):
        if hook_state["vec"] is None:
            return output
        if isinstance(output, tuple):
            h, *rest = output
            h = h.clone()
            h[:, -1, :] = hook_state["vec"].to(h.dtype).to(h.device)
            return (h, *rest)
        h = output.clone()
        h[:, -1, :] = hook_state["vec"].to(h.dtype).to(h.device)
        return h
    handle = model.model.layers[args.steer_layer].register_forward_hook(hook)

    families = args.families.split(",")
    for fam in families:
        print(f"\n{'='*70}\n=== family: {fam} ===\n{'='*70}")
        cents, labels_v, value_tokens, pairs, base_prompts, cyclic = family_config(
            fam, args.steer_layer, labels, pids, X_full, tokenizer)
        print(f"  centroids: {len(cents)} ({labels_v[:3]}...{labels_v[-3:]})")
        print(f"  cyclic={cyclic}")
        print(f"  base prompts: {len(base_prompts)}")
        print(f"  pairs to steer: {pairs}")

        for src_lbl, tgt_lbl in pairs:
            src_idx = labels_v.index(src_lbl); tgt_idx = labels_v.index(tgt_lbl)
            print(f"\n  steering {src_lbl}({src_idx}) → {tgt_lbl}({tgt_idx})...")
            lin_p = linear_path(cents[src_idx], cents[tgt_idx], args.n_waypoints)
            ours_p, ours_idx = cycle_arc_path(cents, src_idx, tgt_idx, args.n_waypoints, cyclic=cyclic)
            paper_p = paper_spline_path(cents, src_idx, tgt_idx, args.n_waypoints, cyclic=cyclic)

            lin_pr = get_path_probs(model, tokenizer, hook_state, args.steer_layer,
                                      lin_p, base_prompts, value_tokens)
            ours_pr = get_path_probs(model, tokenizer, hook_state, args.steer_layer,
                                       ours_p, base_prompts, value_tokens)
            paper_pr = get_path_probs(model, tokenizer, hook_state, args.steer_layer,
                                        paper_p, base_prompts, value_tokens)

            if cyclic:
                # Coverage metric: dominant token is intermediate concept
                def coverage(probs):
                    dom = np.argmax(probs, axis=1)
                    intermediates = set((src_idx + k) % len(labels_v)
                                         for k in range(1, ((tgt_idx - src_idx) % len(labels_v))))
                    return float(np.mean([d in intermediates for d in dom]))
                print(f"    coverage(intermediates): linear={coverage(lin_pr):.3f}  "
                      f"paper={coverage(paper_pr):.3f}  ours={coverage(ours_pr):.3f}")
            else:
                # Line manifolds: expected predicted value should slide src→tgt monotonically
                vals_arr = np.array([float(s) for s in display_labels_for(labels_v)])
                def expected(probs):
                    return (probs * vals_arr[None, :]).sum(axis=1) / (probs.sum(axis=1) + 1e-12)
                # Monotonicity: Spearman corr between waypoint index and E[value]
                from scipy.stats import spearmanr
                expected_lin = expected(lin_pr)
                expected_paper = expected(paper_pr)
                expected_ours = expected(ours_pr)
                rho_lin = spearmanr(np.arange(len(expected_lin)), expected_lin).correlation
                rho_paper = spearmanr(np.arange(len(expected_paper)), expected_paper).correlation
                rho_ours = spearmanr(np.arange(len(expected_ours)), expected_ours).correlation
                print(f"    monotonicity rho(waypoint, E[value]): "
                      f"linear={rho_lin:+.3f}  paper={rho_paper:+.3f}  ours={rho_ours:+.3f}")
                # Save expected-value plot for line manifolds (extra panel)
                fig2, ax = plt.subplots(figsize=(10, 4))
                target_value = float(tgt_lbl); source_value = float(src_lbl)
                ax.axhline(source_value, color="gray", linestyle=":", alpha=0.5,
                           label=f"source={src_lbl}")
                ax.axhline(target_value, color="black", linestyle=":", alpha=0.5,
                           label=f"target={tgt_lbl}")
                ax.plot(expected_lin, "o-", label=f"linear (ρ={rho_lin:+.2f})", color="C0")
                ax.plot(expected_paper, "s-", label=f"paper spline (ρ={rho_paper:+.2f})", color="C1")
                ax.plot(expected_ours, "^-", label=f"ours / SAVE (ρ={rho_ours:+.2f})", color="C2")
                ax.set_xlabel("waypoint along path")
                ax.set_ylabel(f"E[{fam}]")
                ax.set_title(f"Expected predicted {fam} along path: {src_lbl} → {tgt_lbl}")
                ax.legend()
                ax.grid(alpha=0.3)
                fig2.savefig(out_dir / f"{fam}_{src_lbl}_to_{tgt_lbl}_expected.png".replace(" ", "_"),
                              dpi=130, bbox_inches="tight")
                plt.close(fig2)

            # Plot
            fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
            # For line manifolds with many anchor values, sub-sample y-tick labels
            display_labels = labels_v
            if not cyclic and len(labels_v) > 20:
                step = max(1, len(labels_v) // 12)
                tick_idx = list(range(0, len(labels_v), step))
            else:
                tick_idx = list(range(len(labels_v)))
            for ax, probs, name in zip(axes,
                                         [lin_pr, paper_pr, ours_pr],
                                         ["LINEAR (chord)",
                                          "PAPER spline through all centroids",
                                          "OURS — SAVE manifold (piecewise cycle arc)"]):
                im = ax.imshow(probs.T, aspect="auto", cmap="viridis",
                               vmin=0, vmax=max(0.1, probs.max()),
                               extent=[0, args.n_waypoints, len(display_labels) - 0.5, -0.5])
                ax.set_yticks(tick_idx)
                ax.set_yticklabels([display_labels[i] for i in tick_idx], fontsize=7)
                ax.set_ylabel("value")
                ax.set_title(f"{name} — {fam}: {src_lbl} → {tgt_lbl}", fontsize=10)
            axes[2].set_xlabel("waypoint along path")
            fig.colorbar(im, ax=axes, label="prob (over family-tokens, normalized)")
            fig.suptitle(f"Steering: {fam} {src_lbl} → {tgt_lbl}  (Llama-3.1-8B base, L{args.steer_layer})",
                         fontsize=12)
            out_png = out_dir / f"{fam}_{src_lbl}_to_{tgt_lbl}.png".replace(" ", "_")
            fig.savefig(out_png, dpi=130, bbox_inches="tight")
            plt.close(fig)
            np.savez(out_png.with_suffix(".npz"),
                      linear=lin_pr, paper=paper_pr, ours=ours_pr,
                      labels=display_labels)
            print(f"    wrote {out_png.name}")

    handle.remove()
    print(f"\nall done. figures in {out_dir}/")


if __name__ == "__main__":
    main()

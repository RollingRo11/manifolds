"""V5 steering — value-at-end prompt structure for line manifolds.

For each (family, source, target) and each steering layer:
  Linear / Paper-spline / SAVE manifold steering
  At v5 layers: 16, 19, 22, 26, 28 (paper used L28 for steering)

Steering prompt is the *bare* prompt (no completion). Extraction was at
the completion-value position. So injecting at the last position of the
bare prompt puts the value-encoding residual right where the next-token
prediction happens — same alignment as days.
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
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))
from load_acts import load_labels, read_last_token_per_seq

LLAMA = ("/data/artifacts/rohan/santi/hf_cache/hub/models--meta-llama--Llama-3.1-8B/"
         "snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b")
SHARD = Path("/data/artifacts/rohan/manifolds/v5_llama31_8b_base/shard_dp00")
LABELS = Path("/home/rkathuria/manifolds/data/labels_v5.jsonl")
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def linear_path(c_src, c_tgt, K):
    return np.array([(1 - t) * c_src + t * c_tgt for t in np.linspace(0, 1, K)])


def cycle_arc_path(centroids, src_idx, tgt_idx, K, cyclic):
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
    return out


def paper_spline_path(centroids, src_idx, tgt_idx, K, cyclic):
    C = len(centroids)
    if cyclic:
        t_pts = np.arange(C + 1)
        Y = np.vstack([centroids, centroids[:1]])
        spline = CubicSpline(t_pts, Y, bc_type="periodic", axis=0)
        t_start = float(src_idx); t_end = float(tgt_idx)
        if t_end <= t_start: t_end += C
        return spline(np.linspace(t_start, t_end, K) % C)
    else:
        spline = CubicSpline(np.arange(C), centroids, bc_type="natural", axis=0)
        return spline(np.linspace(src_idx, tgt_idx, K))


def get_path_logit_dist(model, tokenizer, hook_state, path,
                          base_prompts, value_token_ids):
    """For each waypoint and base prompt, return prob distribution over the
    given value tokens (normalized within them)."""
    K = len(path); V = len(value_token_ids)
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


def setup_family(family, layer, labels, pids, X, tokenizer):
    if family == "weekday":
        keep = [i for i, p in enumerate(pids) if labels[p].get("family") == "weekday"]
        Xf = X[keep]
        cidx = [WEEKDAY_ORDER.index(labels[pids[i]]["concept"]) for i in keep]
        cents = np.stack([Xf[np.array(cidx) == k].mean(0) for k in range(7)])
        toks = [tokenizer.encode(" " + d, add_special_tokens=False)[0] for d in WEEKDAY_ORDER]
        prompts = ["It's 5pm on", "It's noon on", "It's midnight on",
                   "It's a sunny afternoon on", "It's the late shift on"]
        return cents, list(WEEKDAY_ORDER), toks, [
            ("Monday", "Friday"), ("Tuesday", "Saturday")], prompts, True

    if family in ("temperature", "age", "year"):
        keep = [i for i, p in enumerate(pids) if labels[p].get("family") == family]
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
            toks.append(tok_ids[0])
        cents = np.stack(cents)
        prompts_per = {"temperature": ["Today it's", "It's currently",
                                          "The temperature is", "Outside it's"],
                       "age": ["They are", "He is", "She is"],
                       "year": ["The year was", "It was the year",
                                "In the year"]}
        prompts = prompts_per[family]
        # Pick endpoints + a mid-range pair
        return cents, label_strs, toks, [
            (label_strs[0], label_strs[-1]),
            (label_strs[len(label_strs)//4], label_strs[3*len(label_strs)//4]),
        ], prompts, False

    if family == "color":
        keep = [i for i, p in enumerate(pids) if labels[p].get("family") == "color"]
        Xf = X[keep]
        # Group by concept (color word)
        concepts = [labels[pids[i]]["concept"] for i in keep]
        uniq = sorted(set(concepts))
        cents, label_strs, toks = [], [], []
        for c in uniq:
            mask = np.array(concepts) == c
            tok_ids = tokenizer.encode(" " + c, add_special_tokens=False)
            if not tok_ids: continue
            cents.append(Xf[mask].mean(0))
            label_strs.append(c)
            toks.append(tok_ids[0])
        cents = np.stack(cents)
        # Use a neutral hex code for steering prompt
        prompts = ["The hex code #808080 is for the color"]
        # Pick pairs of distinct colors
        pairs = []
        if "red" in label_strs and "blue" in label_strs:
            pairs.append(("red", "blue"))
        if "green" in label_strs and "red" in label_strs:
            pairs.append(("green", "red"))
        return cents, label_strs, toks, pairs, prompts, False

    raise ValueError(f"unknown family {family}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steer-layers", default="19,28")
    ap.add_argument("--n-waypoints", type=int, default=50)
    ap.add_argument("--families", default="weekday,temperature,age,year,color")
    ap.add_argument("--out-dir", default="/home/rkathuria/manifolds/figures/steer_v5")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    labels = load_labels(LABELS)

    print("loading Llama-3.1-8B base...")
    tokenizer = AutoTokenizer.from_pretrained(LLAMA)
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    summary = []

    for layer in [int(s) for s in args.steer_layers.split(",")]:
        print(f"\n{'#'*70}\n## STEERING LAYER {layer}\n{'#'*70}")
        X_full, pids = read_last_token_per_seq(SHARD, layer)

        hook_state = {"vec": None}
        def hook(_, __, output):
            if hook_state["vec"] is None:
                return output
            if isinstance(output, tuple):
                h, *rest = output; h = h.clone()
                h[:, -1, :] = hook_state["vec"].to(h.dtype).to(h.device)
                return (h, *rest)
            h = output.clone()
            h[:, -1, :] = hook_state["vec"].to(h.dtype).to(h.device)
            return h
        handle = model.model.layers[layer].register_forward_hook(hook)

        for fam in args.families.split(","):
            print(f"\n=== {fam} ===")
            try:
                cents, lbls, toks, pairs, prompts, cyclic = setup_family(
                    fam, layer, labels, pids, X_full, tokenizer)
            except Exception as e:
                print(f"  setup failed: {e}")
                continue
            if not pairs:
                print(f"  no pairs to test, skipping")
                continue
            print(f"  centroids: {len(cents)} ({lbls[:3]}...{lbls[-3:]})")
            print(f"  pairs: {pairs}  cyclic={cyclic}")

            for src_lbl, tgt_lbl in pairs:
                src = lbls.index(src_lbl); tgt = lbls.index(tgt_lbl)
                lin = linear_path(cents[src], cents[tgt], args.n_waypoints)
                ours = cycle_arc_path(cents, src, tgt, args.n_waypoints, cyclic)
                paper = paper_spline_path(cents, src, tgt, args.n_waypoints, cyclic)

                lin_pr = get_path_logit_dist(model, tokenizer, hook_state, lin, prompts, toks)
                ours_pr = get_path_logit_dist(model, tokenizer, hook_state, ours, prompts, toks)
                paper_pr = get_path_logit_dist(model, tokenizer, hook_state, paper, prompts, toks)

                # Metrics
                if cyclic:
                    intermediates = set((src + k) % len(lbls)
                                          for k in range(1, ((tgt - src) % len(lbls))))
                    lin_metric = float(np.mean([np.argmax(p) in intermediates for p in lin_pr]))
                    ours_metric = float(np.mean([np.argmax(p) in intermediates for p in ours_pr]))
                    paper_metric = float(np.mean([np.argmax(p) in intermediates for p in paper_pr]))
                    metric_name = "intermediate-coverage"
                else:
                    if fam == "color":
                        # nominal data — no monotonicity. Use coverage of any concept
                        # other than src or tgt
                        not_endpoint = lambda p: np.argmax(p) not in {src, tgt}
                        lin_metric = float(np.mean([not_endpoint(p) for p in lin_pr]))
                        ours_metric = float(np.mean([not_endpoint(p) for p in ours_pr]))
                        paper_metric = float(np.mean([not_endpoint(p) for p in paper_pr]))
                        metric_name = "off-endpoint-fraction"
                    else:
                        # Lines: monotonicity of E[value]
                        try:
                            vs = np.array([float(s) for s in lbls])
                        except:
                            vs = np.arange(len(lbls), dtype=float)
                        E = lambda p: (p * vs[None]).sum(axis=1) / (p.sum(axis=1) + 1e-12)
                        lin_metric = spearmanr(np.arange(len(lin_pr)), E(lin_pr)).correlation
                        ours_metric = spearmanr(np.arange(len(ours_pr)), E(ours_pr)).correlation
                        paper_metric = spearmanr(np.arange(len(paper_pr)), E(paper_pr)).correlation
                        metric_name = "monotonicity"

                print(f"  {src_lbl}→{tgt_lbl}: linear={lin_metric:+.3f}  "
                      f"paper={paper_metric:+.3f}  ours={ours_metric:+.3f}  "
                      f"({metric_name})")

                summary.append({"layer": layer, "family": fam,
                                 "source": src_lbl, "target": tgt_lbl,
                                 "metric": metric_name,
                                 "linear": float(lin_metric), "paper": float(paper_metric),
                                 "ours": float(ours_metric)})

                # Plot heatmap
                fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
                tick_idx = list(range(0, len(lbls), max(1, len(lbls) // 12))) if len(lbls) > 15 else list(range(len(lbls)))
                for ax, probs, name in zip(axes, [lin_pr, paper_pr, ours_pr],
                                             ["LINEAR", "PAPER spline", "OURS — SAVE manifold"]):
                    im = ax.imshow(probs.T, aspect="auto", cmap="viridis",
                                    vmin=0, vmax=max(0.1, probs.max()),
                                    extent=[0, args.n_waypoints, len(lbls) - 0.5, -0.5])
                    ax.set_yticks(tick_idx)
                    ax.set_yticklabels([lbls[i] for i in tick_idx], fontsize=7)
                    ax.set_title(f"{name} — {fam}: {src_lbl}→{tgt_lbl} (L{layer})", fontsize=10)
                axes[2].set_xlabel("waypoint")
                fig.colorbar(im, ax=axes)
                fig.suptitle(f"v5 steering: {fam} {src_lbl}→{tgt_lbl}, Llama-3.1-8B base, L{layer}\n"
                              f"linear={lin_metric:+.3f}  paper={paper_metric:+.3f}  ours={ours_metric:+.3f}",
                              fontsize=11)
                fig.savefig(out_dir / f"L{layer}_{fam}_{src_lbl}_to_{tgt_lbl}.png".replace(" ", "_"),
                              dpi=120, bbox_inches="tight")
                plt.close(fig)

        handle.remove()

    import json as _json
    (out_dir / "summary.json").write_text(_json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===")
    for r in summary:
        print(f"  L{r['layer']:>2} {r['family']:>11} {r['source']:>10}→{r['target']:>10}  "
              f"lin={r['linear']:+.3f}  paper={r['paper']:+.3f}  ours={r['ours']:+.3f}  ({r['metric']})")


if __name__ == "__main__":
    main()

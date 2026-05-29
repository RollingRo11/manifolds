"""Phase 20: dynamic manifold-clamp steering.

Static steering applies a fixed offset every token. At high magnitudes that
pushes residuals off-manifold by mid-generation, producing safety-soup gibberish.

Dynamic manifold-clamp instead computes the offset PER TOKEN as a function of
the token's current residual:
  score(h)  = h · v_unit                      (current EA-axis projection)
  push(h)   = clamp(target − score, 0, M)     (only push if below target)
  delta(h)  = push * v_unit * α

Tokens already at-or-past the aware target get zero perturbation. Tokens in
non-aware territory get pulled up to the target but not past it. This is a
SOFT CLAMP toward an aware target along the SAVE direction.

Cost per token: 2 dot products + 1 scale. Same wall-clock as static steering.

Comparison vs:
  - static linear chord at same alpha
  - static SAVE-spline at same alpha
  - dynamic manifold-clamp at same alpha
At magnitudes [2, 3, 5] where static methods crash to gibberish.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from scipy.interpolate import CubicSpline

REPO = Path("/home/rkathuria/activation-harvester")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))
from load_acts_multi import read_last_token_multi_rollout
from save_recovery import save_recovery

OLMO_PATH = ("/data/artifacts/rohan/santi/hf_cache/hub/"
             "models--allenai--Olmo-3.1-32B-Think/snapshots")
FORTRESS_SHARD = Path("/data/artifacts/rohan/activation-harvester/fortress_balanced_v1/shard_dp00")
FORTRESS_LABELS = Path("/home/rkathuria/santi/data/probe_train/rollout_labels_fortress_balanced.jsonl")
LAYER = 40


def find_olmo_snapshot():
    return sorted(Path(OLMO_PATH).iterdir())[0]


def normalize(v):
    return v / (np.linalg.norm(v) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--magnitudes", type=float, nargs="+",
                    default=[1.5, 2.0, 3.0, 5.0])
    ap.add_argument("--n-rollouts", type=int, default=10,
                    help="Per (method × prompt) cell.")
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--target-quantile", type=float, default=0.85,
                    help="Quantile of FORTRESS aware activations to use as target score")
    ap.add_argument("--max-push-mult", type=float, default=2.0,
                    help="Cap push magnitude at this many target-score units")
    ap.add_argument("--n-bins", type=int, default=21)
    ap.add_argument("--save-dim", type=int, default=3)
    ap.add_argument("--tp-size", type=int, default=1)
    ap.add_argument("--mem-fraction-static", type=float, default=0.88)
    ap.add_argument("--max-running-requests", type=int, default=32)
    ap.add_argument("--out-dir", default="/home/rkathuria/manifolds/figures/ea_phase2_v20_clamp")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    vec_dir = out_dir / "vectors"; vec_dir.mkdir(exist_ok=True)

    print("=== Build SAVE-spline + clamp parameters ===", flush=True)
    X, keys = read_last_token_multi_rollout(FORTRESS_SHARD, LAYER, position_from_end=0)
    aware_map = {}
    with open(FORTRESS_LABELS) as f:
        for line in f:
            d = json.loads(line)
            aware_map[(int(d["prompt_id"]), int(d["completion_idx"]))] = bool(d["aware"])
    matched = [(i, aware_map[k]) for i, k in enumerate(keys) if k in aware_map]
    idx = np.array([m[0] for m in matched])
    aware = np.array([m[1] for m in matched])
    Xa = X[idx].astype(np.float64)
    mean_a = Xa[aware].mean(0); mean_n = Xa[~aware].mean(0)
    w_fortress = normalize(mean_a - mean_n)

    save_res = save_recovery(Xa, w_fortress, n_slices=10, d_M=args.save_dim)
    V = save_res.save_dirs.astype(np.float64)
    X_mean = Xa.mean(0)
    Xc = Xa - X_mean

    # SAVE-spline waypoint at q50 (matches v13's best result)
    s = Xa @ w_fortress
    K = args.n_bins
    bin_edges = np.quantile(s, np.linspace(0, 1, K + 1))
    bin_idx = np.clip(np.digitize(s, bin_edges[1:-1]), 0, K - 1)
    Z = Xc @ V
    bin_centroids_z = np.stack([Z[bin_idx == k].mean(0) if (bin_idx == k).any()
                                  else np.zeros(args.save_dim) for k in range(K)])
    bin_centroids_amb = bin_centroids_z @ V.T + X_mean
    sp = CubicSpline(np.arange(K), bin_centroids_amb, bc_type="natural", axis=0)
    spline_wp50 = sp(0.5 * (K - 1))
    chord_amb = bin_centroids_amb[K - 1] - bin_centroids_amb[0]
    chord_norm = float(np.linalg.norm(chord_amb))
    chord_unit = chord_amb / chord_norm
    spline_offset = spline_wp50 - bin_centroids_amb[0]
    spline_unit = spline_offset / np.linalg.norm(spline_offset)

    # Clamp parameters: target = quantile of aware-class scores along SAVE_dir_0
    save_dir0 = V[:, 0] / np.linalg.norm(V[:, 0])
    aware_scores = Xa[aware] @ save_dir0
    non_scores = Xa[~aware] @ save_dir0
    target_score = float(np.quantile(aware_scores, args.target_quantile))
    max_push = float(np.abs(target_score - non_scores.mean()) * args.max_push_mult)
    print(f"  ‖chord‖ = {chord_norm:.2f}", flush=True)
    print(f"  SAVE_dir0: aware mean score = {aware_scores.mean():+.2f} (q{int(args.target_quantile*100)}={target_score:+.2f}); "
          f"non-aware mean = {non_scores.mean():+.2f}", flush=True)
    print(f"  clamp target = {target_score:.2f},  max_push = {max_push:.2f}", flush=True)

    # Save direction vectors as safetensors
    method_paths = {}
    for name, vec in [
        ("linear", chord_unit),
        ("spline_q50", spline_unit),
        ("clamp_save_dir0", save_dir0),
    ]:
        path = vec_dir / f"{name}.safetensors"
        save_file({f"layer_{LAYER}": torch.from_numpy(vec).float()}, str(path))
        method_paths[name] = str(path)

    test_prompts_user = [
        "What's the capital of France?",
        "Could you tell me about photosynthesis?",
        "How do I make a paper airplane?",
        "What's a good recipe for pancakes?",
    ]
    snap = find_olmo_snapshot()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(snap))
    def chat_format(user_msg):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False, add_generation_prompt=True)

    inputs = []; rid_meta = {}; manifest = {}

    # Baseline (no steering)
    for pi, p in enumerate(test_prompts_user):
        ids = tokenizer(chat_format(p), add_special_tokens=False)["input_ids"]
        for r in range(args.n_rollouts):
            rid = f"baseline|{pi}|{r}"
            inputs.append({"rid": rid, "input_ids": ids})
            rid_meta[rid] = {"method": "baseline", "magnitude": 0.0,
                              "prompt_idx": pi, "rollout": r}

    # Static methods × magnitudes
    for method in ["linear", "spline_q50"]:
        for mag in args.magnitudes:
            alpha = float(mag * chord_norm)
            for pi, p in enumerate(test_prompts_user):
                ids = tokenizer(chat_format(p), add_special_tokens=False)["input_ids"]
                for r in range(args.n_rollouts):
                    rid = f"{method}|m{mag}|{pi}|{r}"
                    inputs.append({"rid": rid, "input_ids": ids})
                    manifest[rid] = {"steer_path": method_paths[method],
                                       "alpha": alpha, "layer": LAYER}
                    rid_meta[rid] = {"method": method, "magnitude": mag,
                                      "prompt_idx": pi, "rollout": r}

    # Dynamic clamp method × magnitudes
    # Here alpha modulates the push intensity (1.0 = full clamp, smaller = damped)
    for clamp_alpha in [1.0]:
        for mag_label in args.magnitudes:
            # mag_label is just a label here; actual perturbation is dynamic per-token
            for pi, p in enumerate(test_prompts_user):
                ids = tokenizer(chat_format(p), add_special_tokens=False)["input_ids"]
                for r in range(args.n_rollouts):
                    rid = f"clamp|m{mag_label}|{pi}|{r}"
                    inputs.append({"rid": rid, "input_ids": ids})
                    manifest[rid] = {"steer_path": method_paths["clamp_save_dir0"],
                                       "alpha": clamp_alpha,
                                       "layer": LAYER,
                                       "mode": "manifold_clamp",
                                       "target_score": target_score,
                                       "max_push": max_push * mag_label}
                    rid_meta[rid] = {"method": "clamp", "magnitude": mag_label,
                                      "prompt_idx": pi, "rollout": r}
    print(f"\n  total: {len(inputs)} requests", flush=True)
    manifest_path = out_dir / "steer_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    os.environ["HARVEST_STEER_MANIFEST_PATH"] = str(manifest_path)
    os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "harvester_sglang"
    pp = os.environ.get("PYTHONPATH", "")
    extra = str(REPO / "src")
    if extra not in pp.split(":"):
        os.environ["PYTHONPATH"] = f"{extra}:{pp}" if pp else extra

    print("\n=== SGLang generation ===", flush=True)
    import sglang as sgl
    t0 = time.time()
    engine = sgl.Engine(
        model_path=str(snap), tp_size=args.tp_size,
        disable_radix_cache=True, disable_cuda_graph=True,
        chunked_prefill_size=32768,
        max_running_requests=args.max_running_requests,
        mem_fraction_static=args.mem_fraction_static,
        log_level="info",
    )
    print(f"  engine ready in {time.time()-t0:.1f}s", flush=True)

    sampling_params = {"max_new_tokens": args.max_new_tokens,
                        "temperature": 1.0, "top_p": 0.9}
    rids = [x["rid"] for x in inputs]
    input_ids_batch = [x["input_ids"] for x in inputs]
    t0 = time.time()
    out = engine.generate(input_ids=input_ids_batch,
                           sampling_params=sampling_params, rid=rids)
    elapsed = time.time() - t0
    total_new = sum(o.get("meta_info", {}).get("completion_tokens", 0) for o in out)
    print(f"  generation done in {elapsed:.1f}s ({total_new/max(elapsed,1):.0f} tok/s)", flush=True)

    rollouts = []
    for o, x in zip(out, inputs):
        meta = rid_meta[x["rid"]]
        rollouts.append({
            "rid": x["rid"], "method": meta["method"], "magnitude": meta["magnitude"],
            "prompt": test_prompts_user[meta["prompt_idx"]],
            "rollout": meta["rollout"], "generation": o.get("text", ""),
        })
    (out_dir / "cot_rollouts.json").write_text(json.dumps(rollouts, indent=2))
    print(f"  wrote {out_dir}/cot_rollouts.json ({len(rollouts)} rows)", flush=True)
    engine.shutdown()


if __name__ == "__main__":
    main()

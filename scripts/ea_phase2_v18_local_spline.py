"""Phase 18: per-prompt local SPLINE steering.

Generalizes v15 (per-prompt single direction) to per-prompt multi-waypoint
spline. For each test prompt:
  1. Get its L40 residual h_test (last user-prompt token).
  2. Find K=30 nearest FORTRESS prompts (using h_test).
  3. Among those K prompts, separate into aware-rollouts vs non-aware-rollouts
     and bin by their FORTRESS-axis projection score → 7 quantile-bin centroids.
  4. Cubic-spline through these 7 LOCAL centroids → personalized manifold curve.
  5. Sample waypoints at fractions {0, 0.25, 0.5, 0.75, 1.0}.
  6. Use these LOCAL spline waypoints as steering offsets at matched magnitudes.

Compare against:
  - global linear (the v10 chord, FORTRESS-derived)
  - global spline (v10's spline through 21 quantile centroids)
  - local linear (v15's per-prompt single direction)
  - local spline (this new approach)

The hypothesis: local spline through nearest-neighbors should be the most
geometry-respecting per-prompt steering — captures local curvature in the
neighborhood the test prompt actually lives in.
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
                    default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0])
    ap.add_argument("--n-rollouts", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--knn-k", type=int, default=60,
                    help="K nearest FORTRESS rollouts per test prompt (need many for binning)")
    ap.add_argument("--n-bins", type=int, default=7,
                    help="quantile bins for local spline")
    ap.add_argument("--tp-size", type=int, default=1)
    ap.add_argument("--mem-fraction-static", type=float, default=0.88)
    ap.add_argument("--max-running-requests", type=int, default=16)
    ap.add_argument("--out-dir", default="/home/rkathuria/manifolds/figures/ea_phase2_v18_local_spline")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    vec_dir = out_dir / "vectors"; vec_dir.mkdir(exist_ok=True)

    # ===== Phase 1: load FORTRESS data =====
    print("=== Phase 1: load FORTRESS L40 ===", flush=True)
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
    print(f"  {Xa.shape[0]} acts (aware={aware.sum()} non={(~aware).sum()})", flush=True)

    # Global probe direction (for binning the local data)
    mean_a = Xa[aware].mean(0); mean_n = Xa[~aware].mean(0)
    w_global = normalize(mean_a - mean_n)
    chord_norm_global = float(np.linalg.norm(mean_a - mean_n))
    print(f"  global w_fortress ‖={chord_norm_global:.2f}", flush=True)

    # ===== Phase 2: extract test prompt residuals =====
    print("\n=== Phase 2: get test prompt residuals (HF pass) ===", flush=True)
    test_prompts_user = [
        "What's the capital of France?",
        "Could you tell me about photosynthesis?",
        "How do I make a paper airplane?",
        "What's a good recipe for pancakes?",
    ]
    snap = find_olmo_snapshot()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(str(snap))
    def chat_format(user_msg):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False, add_generation_prompt=True)
    test_chats = [chat_format(p) for p in test_prompts_user]
    test_ids = [tokenizer(ct, add_special_tokens=False)["input_ids"] for ct in test_chats]

    print("  loading OLMo for residual extraction...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(snap), dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    test_residuals = []
    captured = {}
    def hook(_, __, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["h"] = h.detach().clone()
    handle = model.model.layers[LAYER].register_forward_hook(hook)
    for i, ids in enumerate(test_ids):
        ids_t = torch.tensor([ids]).cuda()
        with torch.no_grad():
            _ = model(ids_t)
        last_h = captured["h"][0, -1].float().cpu().numpy().astype(np.float64)
        test_residuals.append(last_h)
    handle.remove()
    del model
    torch.cuda.empty_cache()
    test_residuals = np.stack(test_residuals)

    # ===== Phase 3: per-prompt local spline =====
    print(f"\n=== Phase 3: per-prompt local spline (K_nn={args.knn_k}, K_bins={args.n_bins}) ===",
           flush=True)
    methods_per_prompt = {"global_linear": {pi: w_global for pi in range(len(test_prompts_user))}}
    spline_q_fracs = [0.0, 0.25, 0.5, 0.75, 1.0]
    for q_frac in spline_q_fracs[1:]:
        methods_per_prompt[f"local_spline_q{int(q_frac*100)}"] = {}

    for pi, h_test in enumerate(test_residuals):
        # Find K nearest FORTRESS rollouts (regardless of class)
        dists = np.linalg.norm(Xa - h_test, axis=1)
        nn_idx = np.argsort(dists)[: args.knn_k]
        local_X = Xa[nn_idx]
        local_aware = aware[nn_idx]

        # If we have enough of both classes locally, build a local spline
        if local_aware.sum() < 3 or (~local_aware).sum() < 3:
            # Fallback: use global w_fortress
            print(f"  prompt {pi} '{test_prompts_user[pi][:30]}': insufficient local class diversity"
                  f" (aware={local_aware.sum()}, non={(~local_aware).sum()}); using global",
                   flush=True)
            for q_frac in spline_q_fracs[1:]:
                methods_per_prompt[f"local_spline_q{int(q_frac*100)}"][pi] = w_global
            continue

        # Bin the local points by w_global projection (the EA axis)
        local_score = local_X @ w_global
        bin_edges = np.quantile(local_score, np.linspace(0, 1, args.n_bins + 1))
        bin_idx = np.clip(np.digitize(local_score, bin_edges[1:-1]), 0, args.n_bins - 1)

        bin_centroids = []
        for k in range(args.n_bins):
            mask = bin_idx == k
            if mask.sum() < 1:
                # Fall back to nearest filled bin's centroid
                bin_centroids.append(None)
            else:
                bin_centroids.append(local_X[mask].mean(0))
        # Forward-fill any missing
        last_good = None
        for k in range(args.n_bins):
            if bin_centroids[k] is None:
                bin_centroids[k] = last_good if last_good is not None else local_X[0]
            else:
                last_good = bin_centroids[k]
        bin_centroids = np.stack(bin_centroids)

        # Cubic spline through 7 local bin centroids
        sp = CubicSpline(np.arange(args.n_bins), bin_centroids, bc_type="natural", axis=0)
        wp_indices = [f * (args.n_bins - 1) for f in spline_q_fracs]
        waypoints = sp(np.array(wp_indices))      # [5, d]

        for q_frac, wp in zip(spline_q_fracs[1:], waypoints[1:]):
            offset = wp - waypoints[0]
            unit = offset / (np.linalg.norm(offset) + 1e-12)
            methods_per_prompt[f"local_spline_q{int(q_frac*100)}"][pi] = unit

        # Diagnostic
        local_chord = bin_centroids[-1] - bin_centroids[0]
        local_chord_unit = local_chord / np.linalg.norm(local_chord)
        print(f"  prompt {pi} '{test_prompts_user[pi][:30]}': "
              f"local_chord ‖={np.linalg.norm(local_chord):.2f}, "
              f"cos(local_chord, global)={local_chord_unit @ w_global:+.3f}, "
              f"aware_in_NN={local_aware.sum()}/{args.knn_k}", flush=True)

    # ===== Phase 4: build manifest =====
    print(f"\n=== Phase 4: build manifest ===", flush=True)
    inputs = []; rid_meta = {}; manifest = {}
    chord_norm = chord_norm_global  # use global for consistent magnitude scaling

    for pi in range(len(test_prompts_user)):
        for r in range(args.n_rollouts):
            rid = f"baseline|{pi}|{r}"
            inputs.append({"rid": rid, "input_ids": test_ids[pi]})
            rid_meta[rid] = {"method": "baseline", "magnitude": 0.0,
                              "prompt_idx": pi, "rollout": r}

    method_paths = {}
    for method_name, vecs_per_prompt in methods_per_prompt.items():
        for pi, vec in vecs_per_prompt.items():
            path = vec_dir / f"{method_name}_p{pi}.safetensors"
            save_file({f"layer_{LAYER}": torch.from_numpy(vec).float()}, str(path))
            method_paths[(method_name, pi)] = str(path)

    for method_name in methods_per_prompt:
        for mag in args.magnitudes:
            alpha = float(mag * chord_norm)
            for pi in range(len(test_prompts_user)):
                for r in range(args.n_rollouts):
                    rid = f"{method_name}|m{mag}|{pi}|{r}"
                    inputs.append({"rid": rid, "input_ids": test_ids[pi]})
                    manifest[rid] = {"steer_path": method_paths[(method_name, pi)],
                                       "alpha": alpha, "layer": LAYER}
                    rid_meta[rid] = {"method": method_name, "magnitude": mag,
                                      "prompt_idx": pi, "rollout": r}
    print(f"  total: {len(inputs)} requests", flush=True)
    manifest_path = out_dir / "steer_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    os.environ["HARVEST_STEER_MANIFEST_PATH"] = str(manifest_path)
    os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "harvester_sglang"
    pp = os.environ.get("PYTHONPATH", "")
    extra = str(REPO / "src")
    if extra not in pp.split(":"):
        os.environ["PYTHONPATH"] = f"{extra}:{pp}" if pp else extra

    print("\n=== Phase 5: SGLang generation ===", flush=True)
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
    print(f"  wrote {out_dir}/cot_rollouts.json", flush=True)
    engine.shutdown()


if __name__ == "__main__":
    main()

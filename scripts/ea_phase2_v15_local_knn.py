"""Phase 15: prompt-conditional local k-NN steering on FORTRESS.

Hypothesis: each test prompt lives in a specific region of activation space.
A GLOBAL aware-vs-non-aware direction (w_fortress = mean_aware - mean_non) is
the AVERAGE direction across all FORTRESS prompts. For a specific test prompt,
the LOCAL aware-vs-non-aware direction (computed from FORTRESS prompts that
are similar to the test prompt) might give:
  - cleaner steering (more on-task verbalization, less safety-soup)
  - more headroom (coherent at higher magnitudes)
  - more specific behavioral effect

Two-pass approach:
  Pass 1: Get each test prompt's L40 residual via SGLang (no steering).
          Use this to find K=20 nearest FORTRESS prompts.
  Pass 2: Per-test-prompt steering vector = local average of (h_aware - h_non)
          over those K FORTRESS prompts.

Compare vs global w_fortress at matched magnitudes.
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
    ap.add_argument("--knn-k", type=int, default=20,
                    help="K nearest FORTRESS prompts per test prompt")
    ap.add_argument("--tp-size", type=int, default=1)
    ap.add_argument("--mem-fraction-static", type=float, default=0.88)
    ap.add_argument("--max-running-requests", type=int, default=16)
    ap.add_argument("--out-dir", default="/home/rkathuria/manifolds/figures/ea_phase2_v15_local")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    vec_dir = out_dir / "vectors"; vec_dir.mkdir(exist_ok=True)

    # ===== Phase 1: load FORTRESS pair-difference vectors =====
    print("=== Phase 1: load FORTRESS pair vectors ===", flush=True)
    X, keys = read_last_token_multi_rollout(FORTRESS_SHARD, LAYER, position_from_end=0)
    aware_map = {}
    with open(FORTRESS_LABELS) as f:
        for line in f:
            d = json.loads(line)
            aware_map[(int(d["prompt_id"]), int(d["completion_idx"]))] = bool(d["aware"])

    # Build per-prompt pairs: for each pid, h_aware_mean and h_non_mean
    by_pid = {}
    for i, (pid, cidx) in enumerate(keys):
        if (pid, cidx) not in aware_map:
            continue
        is_aware = aware_map[(pid, cidx)]
        by_pid.setdefault(pid, {"aware": [], "non": []})
        bucket = "aware" if is_aware else "non"
        by_pid[pid][bucket].append(X[i].astype(np.float64))

    # Keep prompts that have BOTH classes
    paired_pids = [pid for pid, d in by_pid.items()
                    if d["aware"] and d["non"]]
    pair_h_aware = []
    pair_h_non = []
    pair_delta = []
    pair_h_avg = []   # average of aware + non (for nearest-neighbor lookup)
    for pid in paired_pids:
        h_a = np.stack(by_pid[pid]["aware"]).mean(0)
        h_n = np.stack(by_pid[pid]["non"]).mean(0)
        pair_h_aware.append(h_a)
        pair_h_non.append(h_n)
        pair_delta.append(h_a - h_n)
        pair_h_avg.append((h_a + h_n) / 2)
    pair_h_aware = np.stack(pair_h_aware)
    pair_h_non = np.stack(pair_h_non)
    pair_delta = np.stack(pair_delta)
    pair_h_avg = np.stack(pair_h_avg)
    print(f"  {len(paired_pids)} FORTRESS prompts with both aware/non rollouts", flush=True)

    # Global mean direction (the v10/v12 chord)
    w_global = normalize(pair_delta.mean(0))
    chord_norm_global = float(np.linalg.norm(pair_delta.mean(0)))
    print(f"  global w_fortress ‖={chord_norm_global:.2f}", flush=True)

    # ===== Phase 2: get test prompt residuals (no steering pass) =====
    print("\n=== Phase 2: get test prompt residuals (unsteered SGLang pass) ===",
           flush=True)
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
    test_chats = [chat_format(p) for p in test_prompts_user]
    test_ids = [tokenizer(ct, add_special_tokens=False)["input_ids"] for ct in test_chats]

    # Use HF transformers in float16 for one-time residual extraction (avoid
    # SGLang setup since we just need 4 prompts × 1 forward).
    print("  loading OLMo for residual extraction...", flush=True)
    from transformers import AutoModelForCausalLM
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
        # Last token of user prompt
        last_h = captured["h"][0, -1].float().cpu().numpy().astype(np.float64)
        test_residuals.append(last_h)
        print(f"  test prompt {i} '{test_prompts_user[i][:30]}': L40 residual ‖={np.linalg.norm(last_h):.2f}",
               flush=True)
    handle.remove()
    del model
    torch.cuda.empty_cache()
    test_residuals = np.stack(test_residuals)

    # ===== Phase 3: find local k-NN per test prompt =====
    print(f"\n=== Phase 3: per-prompt local k-NN (K={args.knn_k}) ===", flush=True)
    methods_per_prompt = {}    # method_name -> {prompt_idx -> unit_vec}
    methods_per_prompt["global"] = {pi: w_global for pi in range(len(test_prompts_user))}

    local_vecs = {}
    chord_norms_local = {}
    for pi, h_test in enumerate(test_residuals):
        # Compute distance from h_test to each FORTRESS pair's average
        dists = np.linalg.norm(pair_h_avg - h_test, axis=1)
        nearest_idx = np.argsort(dists)[: args.knn_k]
        local_delta = pair_delta[nearest_idx].mean(0)
        local_norm = np.linalg.norm(local_delta)
        local_unit = local_delta / local_norm
        local_vecs[pi] = local_unit
        chord_norms_local[pi] = float(local_norm)
        print(f"  prompt {pi}: nearest FORTRESS pids {paired_pids[nearest_idx[0]]}, "
              f"{paired_pids[nearest_idx[1]]}... local ‖δ‖={local_norm:.2f}, "
              f"cos(local, global)={local_unit @ w_global:+.3f}", flush=True)
    methods_per_prompt["local_knn"] = local_vecs

    # ===== Phase 4: build per-rid manifest =====
    print(f"\n=== Phase 4: build manifest ===", flush=True)
    inputs = []; rid_meta = {}; manifest = {}

    # For magnitude scaling, use global chord_norm so methods are comparable.
    chord_norm = chord_norm_global

    # Baseline (no steering)
    for pi in range(len(test_prompts_user)):
        ids = test_ids[pi]
        for r in range(args.n_rollouts):
            rid = f"baseline|{pi}|{r}"
            inputs.append({"rid": rid, "input_ids": ids})
            rid_meta[rid] = {"method": "baseline", "magnitude": 0.0,
                              "prompt_idx": pi, "rollout": r}

    # Save per-(method, prompt) vectors as files
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
                ids = test_ids[pi]
                for r in range(args.n_rollouts):
                    rid = f"{method_name}|m{mag}|{pi}|{r}"
                    inputs.append({"rid": rid, "input_ids": ids})
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

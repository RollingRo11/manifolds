"""Phase 21: scaled validation on FORTRESS benign prompts.

Concern: baseline EA verbalisation rate of 20% measured on 4 trivia prompts
might be unrepresentative. Scale up to 100 diverse FORTRESS-benign prompts
to get a credible baseline estimate. Same N=25 rollouts/prompt as v19.

Methods (high-magnitude regime where static methods crash):
  - baseline (no steering)
  - static linear chord at mag=5
  - dynamic manifold-clamp at mag=5

For each rollout, judge:
  - EA verbalisation (Qwen2.5-32B + santi v2 prompt)
  - coherence (Qwen3-Nemotron-32B-GenRM + coherence principle)
  - refusal   (Qwen3-Nemotron-32B-GenRM + refusal principle)

NOTE: prompts are loaded via the datasets library and tokenised in this script.
This script writes them into the chat template for OLMo to process. The
adversarial prompts are NOT read or printed by the driver; we use the BENIGN
column from FORTRESS, which is research-safe paired benign prompts.
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
    ap.add_argument("--n-prompts", type=int, default=100,
                    help="Number of FORTRESS benign prompts to sample")
    ap.add_argument("--n-rollouts", type=int, default=25)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--magnitude", type=float, default=5.0,
                    help="The high-magnitude regime where static crashes")
    ap.add_argument("--target-quantile", type=float, default=0.85)
    ap.add_argument("--max-push-mult", type=float, default=2.0)
    ap.add_argument("--n-bins", type=int, default=21)
    ap.add_argument("--save-dim", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tp-size", type=int, default=1)
    ap.add_argument("--mem-fraction-static", type=float, default=0.88)
    ap.add_argument("--max-running-requests", type=int, default=32)
    ap.add_argument("--out-dir", default="/home/rkathuria/manifolds/figures/ea_phase2_v21_fortress")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    vec_dir = out_dir / "vectors"; vec_dir.mkdir(exist_ok=True)

    # ========== Build SAVE direction + clamp parameters ==========
    print("=== Build SAVE-direction + clamp params from FORTRESS activations ===", flush=True)
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
    s = Xa @ w_fortress

    # SAVE-spline chord
    K = args.n_bins
    bin_edges = np.quantile(s, np.linspace(0, 1, K + 1))
    bin_idx = np.clip(np.digitize(s, bin_edges[1:-1]), 0, K - 1)
    Z = Xc @ V
    bin_centroids_z = np.stack([Z[bin_idx == k].mean(0) if (bin_idx == k).any()
                                  else np.zeros(args.save_dim) for k in range(K)])
    bin_centroids_amb = bin_centroids_z @ V.T + X_mean
    chord_amb = bin_centroids_amb[K - 1] - bin_centroids_amb[0]
    chord_norm = float(np.linalg.norm(chord_amb))
    chord_unit = chord_amb / chord_norm

    # Clamp parameters
    save_dir0 = V[:, 0] / np.linalg.norm(V[:, 0])
    aware_scores = Xa[aware] @ save_dir0
    non_scores = Xa[~aware] @ save_dir0
    target_score = float(np.quantile(aware_scores, args.target_quantile))
    max_push = float(np.abs(target_score - non_scores.mean()) * args.max_push_mult)
    print(f"  ‖chord‖ = {chord_norm:.2f}, target_score = {target_score:.2f}, max_push = {max_push:.2f}",
           flush=True)

    method_paths = {}
    for name, vec in [("linear", chord_unit), ("clamp_dir0", save_dir0)]:
        path = vec_dir / f"{name}.safetensors"
        save_file({f"layer_{LAYER}": torch.from_numpy(vec).float()}, str(path))
        method_paths[name] = str(path)

    # ========== Load FORTRESS benign prompts (no inspection) ==========
    print(f"\n=== Load {args.n_prompts} FORTRESS benign prompts ===", flush=True)
    from datasets import load_dataset
    ds = load_dataset("ScaleAI/fortress_public", cache_dir="/data/artifacts/hf_cache",
                       split="train")
    rng = np.random.default_rng(args.seed)
    n_total = len(ds)
    sampled_idx = rng.choice(n_total, size=min(args.n_prompts, n_total), replace=False)
    sampled_idx = sorted(int(i) for i in sampled_idx)

    # We only use the BENIGN paired prompt, NOT the adversarial one.
    # The prompts are passed directly to OLMo's chat template; we do not log them in plaintext.
    fortress_prompts = []
    for i in sampled_idx:
        row = ds[i]
        # If 'benign_prompt' field empty, fall back to adversarial. But research scope = benign.
        bp = row.get("benign_prompt", None)
        if not bp:
            continue
        fortress_prompts.append({"pid": row.get("ID", i), "text": bp})
    print(f"  loaded {len(fortress_prompts)} benign prompts (IDs sampled with seed={args.seed})",
           flush=True)
    # Save prompt IDs only (no text) for later analysis
    (out_dir / "prompt_ids.json").write_text(
        json.dumps([{"pid": p["pid"], "char_len": len(p["text"])}
                     for p in fortress_prompts], indent=2))

    # ========== Build manifest ==========
    snap = find_olmo_snapshot()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(snap))
    def chat_format(user_msg):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False, add_generation_prompt=True)

    inputs = []; rid_meta = {}; manifest = {}
    methods_to_run = ["baseline", "linear", "clamp"]

    for pi, fp in enumerate(fortress_prompts):
        ids = tokenizer(chat_format(fp["text"]), add_special_tokens=False)["input_ids"]
        # Truncate if absurdly long (safety)
        if len(ids) > 4096:
            ids = ids[-4096:]
        for method in methods_to_run:
            for r in range(args.n_rollouts):
                rid = f"{method}|p{pi}|r{r}"
                inputs.append({"rid": rid, "input_ids": ids})
                rid_meta[rid] = {"method": method,
                                  "pid": fp["pid"], "prompt_idx": pi,
                                  "rollout": r}
                if method == "baseline":
                    continue
                if method == "linear":
                    alpha = float(args.magnitude * chord_norm)
                    manifest[rid] = {"steer_path": method_paths["linear"],
                                       "alpha": alpha, "layer": LAYER}
                elif method == "clamp":
                    manifest[rid] = {"steer_path": method_paths["clamp_dir0"],
                                       "alpha": 1.0, "layer": LAYER,
                                       "mode": "manifold_clamp",
                                       "target_score": target_score,
                                       "max_push": max_push * args.magnitude}
    print(f"\n  total: {len(inputs)} requests "
          f"({len(fortress_prompts)} prompts × {len(methods_to_run)} cells × {args.n_rollouts} rollouts)",
           flush=True)
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
    print(f"  generation done in {elapsed:.1f}s ({total_new/max(elapsed,1):.0f} tok/s)",
           flush=True)

    rollouts = []
    for o, x in zip(out, inputs):
        meta = rid_meta[x["rid"]]
        rollouts.append({
            "rid": x["rid"],
            "method": meta["method"],
            "magnitude": args.magnitude if meta["method"] != "baseline" else 0.0,
            "pid": meta["pid"],
            "prompt_idx": meta["prompt_idx"],
            "rollout": meta["rollout"],
            # store prompt text for judge use
            "prompt": fortress_prompts[meta["prompt_idx"]]["text"],
            "generation": o.get("text", ""),
        })
    (out_dir / "cot_rollouts.json").write_text(json.dumps(rollouts, indent=2))
    print(f"  wrote {out_dir}/cot_rollouts.json ({len(rollouts)} rows)", flush=True)
    engine.shutdown()


if __name__ == "__main__":
    main()

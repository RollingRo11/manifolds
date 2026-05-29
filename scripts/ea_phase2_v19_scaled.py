"""Phase 19: scaled-up linear vs SAVE-spline_q50 head-to-head at mag=1.5.

100 rollouts per method × baseline = 300 total (25 rollouts × 4 prompts).
Same SAVE-3D-denoised cubic spline construction as v13. One magnitude only.
Standard error at N=100 with p~0.25 is ~4.3pp, so a 25%-vs-5% gap is ~5 sigma.
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
    ap.add_argument("--magnitude", type=float, default=1.5)
    ap.add_argument("--n-rollouts", type=int, default=25,
                    help="Per (method × prompt). With 4 prompts → 4×n_rollouts per method.")
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--n-bins", type=int, default=21)
    ap.add_argument("--save-dim", type=int, default=3)
    ap.add_argument("--tp-size", type=int, default=1)
    ap.add_argument("--mem-fraction-static", type=float, default=0.88)
    ap.add_argument("--max-running-requests", type=int, default=32)
    ap.add_argument("--out-dir", default="/home/rkathuria/manifolds/figures/ea_phase2_v19_scaled")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    vec_dir = out_dir / "vectors"; vec_dir.mkdir(exist_ok=True)

    print("=== Build SAVE-spline manifold ===", flush=True)
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
    Z = Xc @ V
    s = Xa @ w_fortress

    K = args.n_bins
    bin_edges = np.quantile(s, np.linspace(0, 1, K + 1))
    bin_idx = np.clip(np.digitize(s, bin_edges[1:-1]), 0, K - 1)
    bin_centroids_z = np.stack([Z[bin_idx == k].mean(0) if (bin_idx == k).any()
                                  else np.zeros(args.save_dim) for k in range(K)])
    bin_centroids_amb = bin_centroids_z @ V.T + X_mean
    sp = CubicSpline(np.arange(K), bin_centroids_amb, bc_type="natural", axis=0)
    spline_wp50 = sp(0.5 * (K - 1))                        # waypoint at 50% along path

    chord_amb = bin_centroids_amb[K - 1] - bin_centroids_amb[0]
    chord_norm = float(np.linalg.norm(chord_amb))
    chord_unit = chord_amb / chord_norm
    spline_offset = spline_wp50 - bin_centroids_amb[0]
    spline_unit = spline_offset / np.linalg.norm(spline_offset)

    print(f"  ‖chord‖ = {chord_norm:.2f}", flush=True)
    print(f"  cos(linear, spline_q50) = {chord_unit @ spline_unit:.4f}", flush=True)

    methods = {"linear": chord_unit, "spline_q50": spline_unit}
    method_paths = {}
    for name, vec in methods.items():
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

    # Baseline
    for pi, p in enumerate(test_prompts_user):
        ids = tokenizer(chat_format(p), add_special_tokens=False)["input_ids"]
        for r in range(args.n_rollouts):
            rid = f"baseline|{pi}|{r}"
            inputs.append({"rid": rid, "input_ids": ids})
            rid_meta[rid] = {"method": "baseline", "magnitude": 0.0,
                              "prompt_idx": pi, "rollout": r}

    for method in methods:
        alpha = float(args.magnitude * chord_norm)
        for pi, p in enumerate(test_prompts_user):
            ids = tokenizer(chat_format(p), add_special_tokens=False)["input_ids"]
            for r in range(args.n_rollouts):
                rid = f"{method}|m{args.magnitude}|{pi}|{r}"
                inputs.append({"rid": rid, "input_ids": ids})
                manifest[rid] = {"steer_path": method_paths[method],
                                   "alpha": alpha, "layer": LAYER}
                rid_meta[rid] = {"method": method, "magnitude": args.magnitude,
                                  "prompt_idx": pi, "rollout": r}
    print(f"\n  total: {len(inputs)} requests "
          f"({args.n_rollouts}/prompt × 4 prompts × 3 cells = {3 * args.n_rollouts * 4})", flush=True)
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

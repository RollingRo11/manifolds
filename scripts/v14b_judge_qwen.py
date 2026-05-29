"""LLM-as-judge using Qwen2.5-32B-Instruct via vLLM, with the existing
santi/llm_judge_aware.py v2 prompt and parser. Reads cot_rollouts.json
(our manifold-steering output format), outputs cot_rollouts_judged.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Pull in santi's prompt + parser
SANTI = Path("/home/rkathuria/santi")
sys.path.insert(0, str(SANTI))
sys.path.insert(0, str(SANTI / "scripts"))

# Import directly to avoid pulling in async/vllm-old machinery
def _load_aware_v2_prompt():
    from llm_judge_aware import SCORER_PROMPT_V2
    return SCORER_PROMPT_V2

def _load_parser():
    from eval import parse_scorer_response
    return parse_scorer_response


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--judge-model",
                    default="/data/artifacts/hf_cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    args = ap.parse_args()

    rollouts = json.loads(args.input.read_text())
    print(f"loaded {len(rollouts)} rollouts from {args.input}", flush=True)

    SCORER_PROMPT_V2 = _load_aware_v2_prompt()
    parse_scorer_response = _load_parser()
    print(f"using SCORER_PROMPT_V2 ({len(SCORER_PROMPT_V2)} chars)", flush=True)

    from transformers import AutoTokenizer
    print(f"loading tokenizer {args.judge_model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.judge_model)

    formatted = []
    for r in rollouts:
        # Use the rollout itself as the "trace" — our generations are already
        # the model's CoT (they're the assistant's output after <think>).
        trace = r["generation"]
        msg = SCORER_PROMPT_V2.replace("{RESPONSE}", trace)
        s = tok.apply_chat_template(
            [{"role": "user", "content": msg}], tokenize=False,
            add_generation_prompt=True)
        formatted.append(s)

    from vllm import LLM, SamplingParams
    print(f"starting vLLM (tp={args.tp})...", flush=True)
    llm = LLM(model=args.judge_model, tensor_parallel_size=args.tp,
               gpu_memory_utilization=args.gpu_mem,
               max_model_len=8192, enable_prefix_caching=True)
    params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    print(f"judging {len(formatted)} rollouts...", flush=True)
    out = llm.generate(formatted, params)
    print(f"done", flush=True)

    judged = []
    n_aware = 0; n_parse_fail = 0
    for r, o in zip(rollouts, out):
        text = o.outputs[0].text.strip()
        aware, reasoning, quote = parse_scorer_response(text)
        if aware is None:
            n_parse_fail += 1
        if aware:
            n_aware += 1
        rec = dict(r)
        rec["aware"] = aware
        rec["reasoning"] = reasoning
        rec["quote"] = quote
        rec["judge_raw"] = text[:200]
        judged.append(rec)

    out_path = args.out or args.input.parent / "cot_rollouts_judged.json"
    out_path.write_text(json.dumps(judged, indent=2))
    print(f"\nwrote {out_path}", flush=True)
    print(f"aware: {n_aware}/{len(judged)} ({100*n_aware/max(len(judged),1):.1f}%)  "
          f"parse_fail: {n_parse_fail}", flush=True)

    # Aggregate by (method, magnitude)
    bucket = defaultdict(list)
    for r in judged:
        bucket[(r["method"], r["magnitude"])].append(r)

    print("\n=== aware rate by (method, magnitude) ===", flush=True)
    print(f"  {'method':>16s} {'mag':>6s}  {'aware/total':>14s}  {'rate':>8s}")
    rows = []
    for (method, mag), rs in sorted(bucket.items()):
        valid = [r for r in rs if r["aware"] is not None]
        if not valid:
            continue
        rate = sum(1 for r in valid if r["aware"]) / len(valid)
        rows.append((method, mag, sum(1 for r in valid if r["aware"]), len(valid), rate))
        print(f"  {method:>16s} {mag:>6.1f}  "
              f"{sum(1 for r in valid if r['aware']):>5d}/{len(valid):<7d}  {rate:>7.2%}")

    if rows:
        best = max(rows, key=lambda r: r[4])
        print(f"\n=== HEADLINE: highest aware rate = {best[0]} mag={best[1]} → {best[4]:.2%}",
               flush=True)


if __name__ == "__main__":
    main()

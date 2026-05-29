"""Build diverse WildChat-based prompt set for unsupervised manifold discovery.

No EA labels, no hand-crafted framings. Just real diverse user prompts plus
a canonical " The answer" completion to anchor the residual at a fixed position.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import zstandard as zstd

SRC = Path("/home/rkathuria/santi/data/prompts_wildchat.jsonl")
OUT_DIR = Path("/home/rkathuria/manifolds/data")


def main():
    out_prompts = OUT_DIR / "wildchat_prompts.jsonl"
    out_completions = OUT_DIR / "wildchat_completions.jsonl.zst"

    rows = []
    with open(SRC) as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            text = d.get("prompt", "").strip()
            if not text or len(text) > 4000:
                continue
            rows.append({"id": i, "prompt": text})

    with open(out_prompts, "w") as f:
        for r in rows:
            f.write(json.dumps({"id": r["id"], "prompt": r["prompt"]}) + "\n")

    cctx = zstd.ZstdCompressor(level=3)
    with open(out_completions, "wb") as f, cctx.stream_writer(f) as w:
        for r in rows:
            line = json.dumps({
                "prompt_id": r["id"],
                "completion_idx": 0,
                "text": " The answer",
            }) + "\n"
            w.write(line.encode("utf-8"))

    print(f"prompts:     {out_prompts}  ({len(rows)} rows)")
    print(f"completions: {out_completions}")


if __name__ == "__main__":
    main()

"""Paper-faithful color manifold prompts.

Template: "The hex code {h_code} is for the color"
The activation is captured at the last token; the completion is the predicted
color word (red, blue, green, etc.).

Sampling strategy: regular grid over RGB cube to give ~900 colors with broad
coverage of the color space. The paper uses n~900.
"""
from __future__ import annotations

import json
from pathlib import Path

import zstandard as zstd


def hex_code(r, g, b):
    return f"#{r:02X}{g:02X}{b:02X}"


def main():
    out_dir = Path("/home/rkathuria/manifolds/data")
    rows = []
    next_id = 0

    # Sample 10x10x10 grid of RGB → 1000 colors
    step = 28  # 10 levels × 28 ≈ 252, gives 0..252 in steps of 28
    for r in range(0, 256, step):
        for g in range(0, 256, step):
            for b in range(0, 256, step):
                hc = hex_code(r, g, b)
                # Coarse color label by RGB max — used only post-hoc, not by the model
                labels = ["red", "green", "blue"]
                vals = [r, g, b]
                if max(vals) - min(vals) < 30:
                    label = "gray"
                else:
                    label = labels[max(range(3), key=lambda i: vals[i])]
                rows.append({
                    "id": next_id, "prompt": f"The hex code {hc} is for the color",
                    "concept": label, "family": "color",
                    "value": [r, g, b], "hex": hc,
                })
                next_id += 1

    import random
    rng = random.Random(0)
    rng.shuffle(rows)
    for new_id, r in enumerate(rows):
        r["id"] = new_id

    prompts_path = out_dir / "prompts_colors.jsonl"
    with open(prompts_path, "w") as f:
        for r in rows:
            f.write(json.dumps({"id": r["id"], "prompt": r["prompt"]}) + "\n")
    labels_path = out_dir / "labels_colors.jsonl"
    with open(labels_path, "w") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("id", "concept", "family",
                                                    "value", "hex")}) + "\n")
    completions_path = out_dir / "completions_colors.jsonl.zst"
    cctx = zstd.ZstdCompressor(level=3)
    with open(completions_path, "wb") as f, cctx.stream_writer(f) as w:
        for r in rows:
            w.write((json.dumps({"prompt_id": r["id"], "completion_idx": 0,
                                  "text": " " + r["concept"]}) + "\n").encode())
    print(f"prompts: {prompts_path}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()

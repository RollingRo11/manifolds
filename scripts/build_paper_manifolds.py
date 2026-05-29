"""Paper-faithful prompts for additional manifold types from Wurgaft et al.

  temperature (line):  "Today it's {f} degrees Fahrenheit outside"     n=150
  age         (line):  "They are {age} years old."                     n=99
  years       (helix): "The date is year {y}"                          n=199

For SAVE/v1 we capture the activation at the last *non-value* token of
each prompt. The value is the manifold coordinate (concept label).
"""
from __future__ import annotations

import io
import json
import random
from pathlib import Path

import zstandard as zstd

# Temperature: integer degrees F from -20 to 129
TEMPERATURES = list(range(-20, 130))   # 150 values
# Age: 0 .. 98
AGES = list(range(0, 99))              # 99 values
# Years: 1825 .. 2023
YEARS = list(range(1825, 2024))        # 199 values


def main():
    rng = random.Random(0)
    out_dir = Path("/home/rkathuria/manifolds/data")

    rows = []
    next_id = 0

    for f in TEMPERATURES:
        rows.append({"id": next_id, "prompt": f"Today it's {f} degrees Fahrenheit",
                     "concept": str(f), "family": "temperature", "value": f})
        next_id += 1
    for age in AGES:
        rows.append({"id": next_id, "prompt": f"They are {age} years",
                     "concept": str(age), "family": "age", "value": age})
        next_id += 1
    for y in YEARS:
        rows.append({"id": next_id, "prompt": f"The date is year",
                     "concept": str(y), "family": "year", "value": y})
        next_id += 1

    # For years template the variable is captured at completion (the year token
    # we'll feed via completion text), matching the paper's "The date is year".
    # Same for the other two — we feed the value as the completion text.

    rng.shuffle(rows)
    for new_id, r in enumerate(rows):
        r["id"] = new_id

    prompts_path = out_dir / "prompts_v4_paper.jsonl"
    with open(prompts_path, "w") as f:
        for r in rows:
            f.write(json.dumps({"id": r["id"], "prompt": r["prompt"]}) + "\n")
    labels_path = out_dir / "labels_v4_paper.jsonl"
    with open(labels_path, "w") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("id", "concept", "family", "value")}) + "\n")
    completions_path = out_dir / "completions_v4_paper.jsonl.zst"
    cctx = zstd.ZstdCompressor(level=3)
    with open(completions_path, "wb") as f, cctx.stream_writer(f) as w:
        for r in rows:
            line = json.dumps({"prompt_id": r["id"], "completion_idx": 0,
                               "text": " " + str(r["value"])}) + "\n"
            w.write(line.encode("utf-8"))
    print(f"prompts: {prompts_path}  ({len(rows)} rows)")
    by_fam = {}
    for r in rows:
        by_fam[r["family"]] = by_fam.get(r["family"], 0) + 1
    print("  by family:", by_fam)


if __name__ == "__main__":
    main()

"""Paper-faithful prompt set for the days manifold.

Template: "It's {time} on {day}"  (exactly the paper's days template,
where {time} is a small contextual phrase and {day} is the target weekday
captured as the last token of the prompt+completion pair).

n ≈ 60 times × 7 days = 420 prompts, matching the paper.
"""
from __future__ import annotations

import io
import json
import random
from pathlib import Path

import zstandard as zstd

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# 60 short context phrases — minimal variation, mostly in time references
TIMES = [
    "5pm", "9am", "noon", "midnight", "3am", "6pm", "8am", "1pm", "10pm",
    "midmorning", "mid-afternoon", "early morning", "late evening",
    "around lunch", "after dinner", "before breakfast", "right after sunrise",
    "just past sunset", "the late shift", "the early shift",
    "a quiet morning", "a busy afternoon", "a slow evening", "a rainy day",
    "a sunny afternoon", "a cold morning", "a hot afternoon",
    "the first hour", "the last hour", "the lunch hour",
    "the rush hour", "the off hour", "the breakfast hour",
    "happy hour", "early lunchtime", "late lunchtime",
    "a peaceful morning", "a hectic afternoon", "a calm evening",
    "a foggy morning", "a clear morning", "a windy afternoon",
    "the early shift", "the night shift", "the dawn shift",
    "first thing", "last thing", "breakfast time", "lunch time", "dinner time",
    "tea time", "coffee time", "snack time", "story time",
    "study time", "rest time", "play time", "work time",
    "the wee hours", "the witching hour", "twilight",
]

# Months — same idea, different template
MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]

MONTH_CONTEXTS = [
    "the first week", "the last week", "early on", "late in", "midway through",
    "the middle of", "the end of", "the start of", "halfway through",
    "the third week of", "the second week of", "the fourth week of",
    "a chilly morning in", "a sunny day in", "a rainy day in",
    "a warm afternoon in", "a cool evening in", "a bright morning in",
    "the holiday in", "the weekend in", "the weekday in",
    "the start of school in", "the end of school in",
    "the harvest in", "the planting season in", "the rainy season in",
    "an ordinary day in", "a special day in", "a quiet day in",
    "the busiest week of", "the slowest week of",
    "a public holiday in", "a school break in",
    "the convention in", "the fair in", "the festival in",
    "the conference in", "the workshop in", "the seminar in",
    "her birthday is in", "his anniversary is in",
    "their wedding was in", "the meeting was in",
    "the trip was scheduled for", "the launch was set for",
    "the deadline fell in", "the review happens in",
    "the audit is due in", "the report is due in",
    "the next session is in", "the previous session was in",
    "early days of", "the closing days of",
    "summer arrived early in", "winter arrived late in",
    "spring began in", "autumn ended in",
    "the long weekend in", "the short break in",
]


def build_days(rng):
    rows = []
    for di, day in enumerate(WEEKDAYS):
        for ti, time in enumerate(TIMES):
            text = f"It's {time} on"
            rows.append({"id": len(rows), "prompt": text,
                          "concept": day, "family": "weekday",
                          "context_idx": ti})
    rng.shuffle(rows)
    for new_id, r in enumerate(rows):
        r["id"] = new_id
    return rows


def build_months(rng):
    rows = []
    for mi, month in enumerate(MONTHS):
        for ci, ctx in enumerate(MONTH_CONTEXTS):
            text = f"It was {ctx}"
            rows.append({"id": len(rows), "prompt": text,
                          "concept": month, "family": "month",
                          "context_idx": ci})
    rng.shuffle(rows)
    return rows


def main():
    rng = random.Random(0)
    out_dir = Path("/home/rkathuria/manifolds/data")
    weekday_rows = build_days(rng)
    month_rows = build_months(rng)
    # Reassign ids globally
    rows = weekday_rows + month_rows
    for new_id, r in enumerate(rows):
        r["id"] = new_id

    prompts_path = out_dir / "prompts_v3.jsonl"
    with open(prompts_path, "w") as f:
        for r in rows:
            f.write(json.dumps({"id": r["id"], "prompt": r["prompt"]}) + "\n")
    labels_path = out_dir / "labels_v3.jsonl"
    with open(labels_path, "w") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("id", "concept", "family", "context_idx")}) + "\n")
    completions_path = out_dir / "completions_v3.jsonl.zst"
    cctx = zstd.ZstdCompressor(level=3)
    with open(completions_path, "wb") as f, cctx.stream_writer(f) as w:
        for r in rows:
            line = json.dumps({"prompt_id": r["id"], "completion_idx": 0,
                               "text": " " + r["concept"]}) + "\n"
            w.write(line.encode("utf-8"))
    print(f"prompts: {prompts_path}  ({len(rows)} rows)")
    print(f"  weekdays: {len(weekday_rows)}  months: {len(month_rows)}")


if __name__ == "__main__":
    main()

"""V5 paper-faithful prompts with value-at-end structure for all manifolds.

Days already worked (value at end). For temperature/age/year, we restructure
so the value comes at the end of the prompt+completion, matching the days
pattern. This puts extraction position and steering injection position in
proper alignment.

Also includes colors paraboloid for the first time.
"""
from __future__ import annotations

import io
import json
import random
from pathlib import Path

import zstandard as zstd

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
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

# Temperature: short prompt, value at end, multiple paraphrases
TEMP_PROMPTS = ["Today it's", "It's currently", "The temperature is",
                "Right now it's", "Outside it's"]
TEMPERATURES = list(range(-20, 130))  # 150 values

# Age: short prompt, value at end
AGE_PROMPTS = ["They are", "He is", "She is", "The person is"]
AGES = list(range(0, 99))

# Year: short prompt, value at end
YEAR_PROMPTS = ["The year was", "It was the year", "In the year",
                "The date was the year"]
YEARS = list(range(1825, 2024))

# Colors: hex → color word. Hex code in middle, color word at end.
RGB_GRID_STEP = 28  # 10x10x10 = 1000 colors
COLOR_PROMPTS = ["The hex code {hc} is for the color",
                 "Hex {hc} represents the color",
                 "The color of #{hc} is"]


def main():
    rng = random.Random(0)
    out_dir = Path("/home/rkathuria/manifolds/data")

    rows = []
    next_id = 0

    # Days — keep structure from v3 but use a smaller subset for harvesting speed
    rng_d = random.Random(1)
    times_subset = rng_d.sample(TIMES, 30)  # 30 paraphrases × 7 = 210
    for di, day in enumerate(WEEKDAYS):
        for time in times_subset:
            rows.append({"id": next_id, "prompt": f"It's {time} on",
                          "concept": day, "family": "weekday",
                          "value": di, "completion_text": " " + day})
            next_id += 1

    # Temperature — short prompts, value at end completion
    rng_t = random.Random(2)
    temp_subset = rng_t.sample(TEMPERATURES, 100)  # downsample to 100 values
    for f in temp_subset:
        for tp in TEMP_PROMPTS:
            rows.append({"id": next_id, "prompt": tp,
                          "concept": str(f), "family": "temperature",
                          "value": f, "completion_text": " " + str(f)})
            next_id += 1

    # Age
    rng_a = random.Random(3)
    age_subset = rng_a.sample(AGES, 75)
    for age in age_subset:
        for ap in AGE_PROMPTS:
            rows.append({"id": next_id, "prompt": ap,
                          "concept": str(age), "family": "age",
                          "value": age, "completion_text": " " + str(age)})
            next_id += 1

    # Year
    rng_y = random.Random(4)
    year_subset = rng_y.sample(YEARS, 75)
    for y in year_subset:
        for yp in YEAR_PROMPTS:
            rows.append({"id": next_id, "prompt": yp,
                          "concept": str(y), "family": "year",
                          "value": y, "completion_text": " " + str(y)})
            next_id += 1

    # Colors — hex code in prompt, color name as completion
    color_word_for_rgb = lambda r, g, b: (
        "gray" if max(r, g, b) - min(r, g, b) < 30
        else ["red", "green", "blue"][max(range(3), key=lambda i: (r, g, b)[i])]
    )
    color_count = 0
    for r in range(0, 256, RGB_GRID_STEP):
        for g in range(0, 256, RGB_GRID_STEP):
            for b in range(0, 256, RGB_GRID_STEP):
                hc = f"#{r:02X}{g:02X}{b:02X}"
                color = color_word_for_rgb(r, g, b)
                # only one prompt template per color to keep harvest manageable
                cp = COLOR_PROMPTS[0].format(hc=hc)
                rows.append({"id": next_id, "prompt": cp,
                              "concept": color, "family": "color",
                              "value": [r, g, b], "completion_text": " " + color})
                next_id += 1
                color_count += 1

    rng.shuffle(rows)
    for new_id, r in enumerate(rows):
        r["id"] = new_id

    prompts_path = out_dir / "prompts_v5.jsonl"
    with open(prompts_path, "w") as f:
        for r in rows:
            f.write(json.dumps({"id": r["id"], "prompt": r["prompt"]}) + "\n")
    labels_path = out_dir / "labels_v5.jsonl"
    with open(labels_path, "w") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("id", "concept", "family", "value")}) + "\n")
    completions_path = out_dir / "completions_v5.jsonl.zst"
    cctx = zstd.ZstdCompressor(level=3)
    with open(completions_path, "wb") as f, cctx.stream_writer(f) as w:
        for r in rows:
            w.write((json.dumps({"prompt_id": r["id"], "completion_idx": 0,
                                  "text": r["completion_text"]}) + "\n").encode())

    print(f"prompts: {prompts_path}  ({len(rows)} rows)")
    by_fam = {}
    for r in rows:
        by_fam[r["family"]] = by_fam.get(r["family"], 0) + 1
    print("  by family:", by_fam)


if __name__ == "__main__":
    main()

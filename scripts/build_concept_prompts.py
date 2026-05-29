"""Generate paraphrased prompts whose natural next token is a target concept word.

Two concept families: weekdays (7) and months (12). Each concept gets K paraphrases.
We write prompts.jsonl (id, prompt, concept, family) and completions.jsonl.zst with
text == " <concept>" so the harvester captures the activation at the concept token
position (the last token of prompt+completion).

Output is mixed across concepts; the `concept` and `family` fields are kept as
ground truth labels we will hide during unsupervised manifold discovery and only
read back for validation.
"""
from __future__ import annotations

import io
import json
import random
from pathlib import Path

import zstandard as zstd

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]

# Templates that elicit a weekday token. {prev}, {next}, {today} placeholders.
WEEKDAY_TEMPLATES = [
    "The day after {prev} is",
    "The day before {next} is",
    "Today is Wednesday. Yesterday was Tuesday. Tomorrow will be",  # neutral, replaced below
    "If yesterday was {prev}, today is",
    "If tomorrow is {next}, today is",
    "Two days after {prev2} is",
    "Two days before {next2} is",
    "After {prev}, the next day is",
    "Before {next}, the previous day is",
    "Between {prev} and {next} comes",
    "{prev} is followed by",
    "{next} is preceded by",
    "Q: What day comes right after {prev}? A:",
    "Q: What day comes right before {next}? A:",
    "On a calendar, the day immediately after {prev} is",
    "Three days after {prev3} is",
    "Three days before {next3} is",
    "She works {prev}, then takes off the next day, which is",
    "He arrives on {prev}, leaves the following day:",
    "First {prev}, then comes",
    "Just before {next}, on",
    "The weekday following {prev} is called",
    "Question: name the day after {prev}. Answer:",
    "Question: name the day before {next}. Answer:",
    "Calendar order: ..., {prev},",
    "Calendar order: ..., {prev2}, {prev},",
    "He told her the meeting is on the day after {prev}, which is",
    "The standard week order goes ..., {prev},",
]

MONTH_TEMPLATES = [
    "The month after {prev} is",
    "The month before {next} is",
    "If last month was {prev}, this month is",
    "If next month is {next}, this month is",
    "Two months after {prev2} is",
    "Two months before {next2} is",
    "After {prev}, the next month is",
    "Before {next}, the previous month is",
    "Between {prev} and {next} comes",
    "{prev} is followed by",
    "{next} is preceded by",
    "Q: What month comes right after {prev}? A:",
    "Q: What month comes right before {next}? A:",
    "On a calendar, the month immediately after {prev} is",
    "Three months after {prev3} is",
    "Three months before {next3} is",
    "First {prev}, then comes",
    "Just before {next}, in",
    "The month following {prev} is called",
    "Question: name the month after {prev}. Answer:",
    "Question: name the month before {next}. Answer:",
    "Calendar order: ..., {prev},",
    "Calendar order: ..., {prev2}, {prev},",
    "Twelve months in a year: ..., {prev},",
    "She was born one month after {prev}, in",
    "His birthday is the month right after {prev}, namely",
]


def _cycle(items: list[str], idx: int, offset: int) -> str:
    return items[(idx + offset) % len(items)]


def render(template: str, items: list[str], idx: int) -> str:
    repl = {
        "prev": _cycle(items, idx, -1),
        "next": _cycle(items, idx, +1),
        "prev2": _cycle(items, idx, -2),
        "next2": _cycle(items, idx, +2),
        "prev3": _cycle(items, idx, -3),
        "next3": _cycle(items, idx, +3),
        "today": items[idx],
    }
    return template.format(**repl)


def gen(items: list[str], templates: list[str], family: str, start_id: int, rng: random.Random):
    rows = []
    for ci, concept in enumerate(items):
        for ti, tpl in enumerate(templates):
            if "{today}" in tpl or "Wednesday" in tpl:  # skip the dummy template
                if "Tomorrow will be" in tpl and concept != "Thursday":
                    continue
            try:
                txt = render(tpl, items, ci)
            except Exception:
                continue
            rows.append({
                "id": start_id + len(rows),
                "prompt": txt,
                "concept": concept,
                "family": family,
                "template_idx": ti,
            })
    rng.shuffle(rows)
    # reassign ids after shuffle so the harvester sees a uniform random order
    for new_id, r in enumerate(rows):
        r["id"] = start_id + new_id
    return rows


def main():
    rng = random.Random(0)
    out_dir = Path("/home/rkathuria/manifolds/data")
    out_dir.mkdir(parents=True, exist_ok=True)

    weekday_rows = gen(WEEKDAYS, WEEKDAY_TEMPLATES, "weekday", 0, rng)
    month_rows = gen(MONTHS, MONTH_TEMPLATES, "month", len(weekday_rows), rng)
    rows = weekday_rows + month_rows

    prompts_path = out_dir / "prompts.jsonl"
    with open(prompts_path, "w") as f:
        for r in rows:
            # Keep the public prompts.jsonl free of labels (concept/family/template_idx)
            # so the harvester can't see them. We'll save labels separately.
            f.write(json.dumps({"id": r["id"], "prompt": r["prompt"]}) + "\n")

    labels_path = out_dir / "labels.jsonl"
    with open(labels_path, "w") as f:
        for r in rows:
            f.write(json.dumps({
                "id": r["id"],
                "concept": r["concept"],
                "family": r["family"],
                "template_idx": r["template_idx"],
            }) + "\n")

    completions_path = out_dir / "completions.jsonl.zst"
    cctx = zstd.ZstdCompressor(level=3)
    with open(completions_path, "wb") as f, cctx.stream_writer(f) as w:
        for r in rows:
            line = json.dumps({
                "prompt_id": r["id"],
                "completion_idx": 0,
                "text": " " + r["concept"],
            }) + "\n"
            w.write(line.encode("utf-8"))

    print(f"prompts: {prompts_path}  ({len(rows)} rows)")
    print(f"labels:  {labels_path}")
    print(f"compl:   {completions_path}")
    print(f"by family: weekdays={len(weekday_rows)} months={len(month_rows)}")


if __name__ == "__main__":
    main()

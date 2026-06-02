"""Paper-faithful prompt set for the Letters (C-Z) and Ages (1-99) manifolds.

Same construction as build_paper_prompts.py (days/months): each prompt is a
short carrier phrase, and the target concept is appended as the completion so
the harvested last token IS the concept token. Layers/model match the
days+months harvest so the manifolds are directly comparable.

Writes (into data/):
  prompts_v6_letters_ages.jsonl       {id, prompt}
  labels_v6_letters_ages.jsonl        {id, concept, family, context_idx}
  completions_v6_letters_ages.jsonl.zst  {prompt_id, completion_idx, text}
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import zstandard as zstd

LETTERS = [chr(c) for c in range(ord("C"), ord("Z") + 1)]   # C..Z  (paper)
AGES = list(range(1, 100))                                  # 1..99 (paper)

# Carrier phrases ending right before the target token. Variation lives here,
# mirroring the days/months TIMES/MONTH_CONTEXTS lists.
LETTER_CONTEXTS = [
    "The next letter is", "Say the letter", "Write down the letter",
    "The letter on the card is", "Point to the letter", "The chosen letter is",
    "Circle the letter", "The highlighted letter is", "Read the letter",
    "The flashcard shows the letter", "She wrote the letter",
    "He pointed at the letter", "The answer is the letter",
    "On the board was the letter", "The tile showed the letter",
    "The kid picked the letter", "The label was the letter",
    "Underline the letter", "The first key was the letter",
    "The sign displayed the letter", "Print the letter",
    "The grade given was the letter", "Type the letter",
    "The marker landed on the letter", "The result was the letter",
]

AGE_CONTEXTS = [
    "She just turned", "He is now", "The patient is aged",
    "The child is", "My grandmother is", "The form says age",
    "He celebrated turning", "The record lists her age as",
    "The boy is exactly", "The applicant is", "Their age is",
    "The candle count was",
]


def build_family(rng, family, values, contexts):
    rows = []
    for vi, val in enumerate(values):
        for ci, ctx in enumerate(contexts):
            rows.append({"id": len(rows), "prompt": ctx,
                         "concept": str(val), "family": family,
                         "context_idx": ci})
    return rows


def main():
    rng = random.Random(0)
    out = Path("/home/rkathuria/manifolds/data")
    rows = (build_family(rng, "letter", LETTERS, LETTER_CONTEXTS)
            + build_family(rng, "age", AGES, AGE_CONTEXTS))
    rng.shuffle(rows)
    for new_id, r in enumerate(rows):
        r["id"] = new_id

    (out / "prompts_v6_letters_ages.jsonl").write_text(
        "".join(json.dumps({"id": r["id"], "prompt": r["prompt"]}) + "\n" for r in rows))
    (out / "labels_v6_letters_ages.jsonl").write_text(
        "".join(json.dumps({k: r[k] for k in ("id", "concept", "family", "context_idx")}) + "\n"
                for r in rows))
    cctx = zstd.ZstdCompressor(level=3)
    with open(out / "completions_v6_letters_ages.jsonl.zst", "wb") as f, cctx.stream_writer(f) as w:
        for r in rows:
            w.write((json.dumps({"prompt_id": r["id"], "completion_idx": 0,
                                 "text": " " + r["concept"]}) + "\n").encode())

    n_letter = sum(r["family"] == "letter" for r in rows)
    n_age = sum(r["family"] == "age" for r in rows)
    print(f"wrote {len(rows)} prompts  (letters: {n_letter}  ages: {n_age})")
    print(f"  letters: {len(LETTERS)} concepts x {len(LETTER_CONTEXTS)} contexts")
    print(f"  ages:    {len(AGES)} concepts x {len(AGE_CONTEXTS)} contexts")


if __name__ == "__main__":
    main()

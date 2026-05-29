"""Scaled-up paraphrase generator for weekday/month prompts.

We compose three slots:
  PREFIX (chatty / formal / bare) × STEM (the concept-eliciting phrasing) × SUFFIX
to multiply the diversity of contexts each concept appears in. Stems are written
once per family; relative position is parameterized via {prev}/{next}/etc.

Aim: ~150-200 templates × C concepts × small filtering = several thousand
prompts per family. Olmo prefill on this is ~30 seconds.
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

# Stems whose natural completion is the {today} weekday/month token.
WEEKDAY_STEMS = [
    "The day after {prev} is",
    "The day before {next} is",
    "If yesterday was {prev}, today is",
    "If tomorrow is {next}, today is",
    "Two days after {prev2} is",
    "Two days before {next2} is",
    "Three days after {prev3} is",
    "Three days before {next3} is",
    "After {prev}, the next day is",
    "Before {next}, the previous day is",
    "Between {prev} and {next} comes",
    "{prev} is followed by",
    "{next} is preceded by",
    "Q: What day comes right after {prev}? A:",
    "Q: What day comes right before {next}? A:",
    "On a calendar, the day immediately after {prev} is",
    "Calendar order: ..., {prev2}, {prev},",
    "Calendar order: ..., {prev3}, {prev2}, {prev},",
    "She works {prev}, then takes off the next day, which is",
    "He arrives on {prev}, leaves the following day:",
    "First {prev}, then comes",
    "Just before {next}, on",
    "The weekday following {prev} is called",
    "Question: name the day after {prev}. Answer:",
    "Question: name the day before {next}. Answer:",
    "He told her the meeting is on the day after {prev}, which is",
    "The standard week order goes ..., {prev},",
    "After Sunday, Monday. After Monday, Tuesday. After {prev},",
    "Mon, Tue, Wed... continuing from {prev}, next is",
    "Following the pattern Mon→Tue→Wed, after {prev} comes",
    "The day labelled '{prev}+1' on the calendar is",
    "She woke up the morning after {prev}, which is",
    "Her shift on {prev} ends at midnight; the next shift falls on",
    "He travels Mon-Fri and rests {prev_long_text}; today after {prev} is",
    "If the calendar reads {prev}, tomorrow it will read",
    "The day following {prev} is",
    "After {prev} comes",
    "Yesterday: {prev}. Today:",
    "Tomorrow: {next}. Today:",
    "Day-of-week sequence: {prev2}, {prev},",
    "{prev}'s successor is",
    "{next}'s predecessor is",
    "Going forward one day from {prev} lands on",
    "Going backward one day from {next} lands on",
    "He scheduled the call for the day after {prev}, namely",
    "She rescheduled to the day before {next}, namely",
    "Drop the appointment on the day after {prev}, that's",
    "Move the meeting to the day before {next}, that's",
    "On {prev} we rest; on the next day, namely,",
    "The press conference is on the day right after {prev}:",
    "The launch is the day right before {next}:",
    "Naming the day immediately after {prev}:",
    "Naming the day immediately before {next}:",
    "The seven-day cycle ..., {prev2}, {prev},",
    "If today is taken to be one day past {prev}, today is",
    "If today is taken to be one day before {next}, today is",
    "Today falls between {prev} and {next}; today is",
    "After Saturday comes Sunday; after Sunday, Monday; after {prev},",
    "The answer to 'what comes after {prev}?' is",
    "The answer to 'what comes before {next}?' is",
    "Print the day after {prev}:",
    "Print the day before {next}:",
    "Output: day after {prev} =",
    "Output: day before {next} =",
    "If we label today as 'd' and yesterday as {prev}, then d =",
    "If d-1 = {prev}, then d =",
    "If d+1 = {next}, then d =",
    "It was {prev}; the next day was",
    "It will be {next}; the day before is",
    "The seven days are Sun, Mon, Tue, Wed, Thu, Fri, Sat; right after {prev} comes",
    "On the calendar wheel, {prev}'s neighbor (clockwise) is",
    "On the calendar wheel, {next}'s neighbor (counter-clockwise) is",
    "Day index +1 from {prev}:",
    "Day index -1 from {next}:",
    "Coming up next after {prev}:",
    "Just before {next}, we have",
    "Sandwich filling: {prev} on one side, on the other side is",
    "Looking at the weekly schedule: after {prev}, before {next}, sits",
    "Q: Following {prev}, what day? A:",
    "Q: Preceding {next}, what day? A:",
    "It's the morning after {prev}, so today must be",
    "It's the night before {next}, so today is",
    "From {prev}, advance one day to",
    "From {next}, retreat one day to",
    "On weekday {prev}+1:",
    "On weekday {next}-1:",
    "If {prev} just passed, today is",
    "If {next} is right around the corner tomorrow, today is",
    "Continuing the sequence Sun→Mon→Tue: after {prev} the next is",
    "Continuing the sequence backwards Sat→Fri→Thu: before {next} the previous is",
    "The day for which {prev} is the previous day:",
    "The day for which {next} is the next day:",
    "Sequence ..., {prev2}, {prev}, ?, {next}, ... fill in:",
    "Saying it plainly: the day after {prev} is",
    "Plainly: the day before {next} is",
    "He counted forward one day from {prev} and arrived at",
    "She counted backward one day from {next} and arrived at",
    "Spelling it out: the day after {prev} =",
    "Spelling it out: the day before {next} =",
    "By convention, {prev} is followed by",
    "By convention, {next} is preceded by",
    "The next entry in the weekday sequence after {prev}:",
    "The previous entry in the weekday sequence before {next}:",
    "Right after {prev} on the weekly cycle is",
    "Right before {next} on the weekly cycle is",
    "His birthday falls on the day after {prev}, which is",
    "Her appointment falls on the day before {next}, namely",
    "Mark the day after {prev} on the calendar:",
    "Mark the day before {next} on the calendar:",
    "We meet the day after {prev}, that is",
    "We met the day before {next}, that is",
    "Weekly planner — after {prev}, plan for",
    "Weekly planner — before {next}, plan for",
    "Count one day forward from {prev}:",
    "Count one day backward from {next}:",
]

MONTH_STEMS = [
    "The month after {prev} is",
    "The month before {next} is",
    "If last month was {prev}, this month is",
    "If next month is {next}, this month is",
    "Two months after {prev2} is",
    "Two months before {next2} is",
    "Three months after {prev3} is",
    "Three months before {next3} is",
    "After {prev}, the next month is",
    "Before {next}, the previous month is",
    "Between {prev} and {next} comes",
    "{prev} is followed by",
    "{next} is preceded by",
    "Q: What month comes right after {prev}? A:",
    "Q: What month comes right before {next}? A:",
    "On a calendar, the month immediately after {prev} is",
    "Calendar order: ..., {prev2}, {prev},",
    "Calendar order: ..., {prev3}, {prev2}, {prev},",
    "First {prev}, then comes",
    "Just before {next}, in",
    "The month following {prev} is called",
    "Question: name the month after {prev}. Answer:",
    "Question: name the month before {next}. Answer:",
    "Twelve months in a year: ..., {prev},",
    "She was born one month after {prev}, in",
    "His birthday is the month right after {prev}, namely",
    "After January comes February. After February comes March. After {prev} comes",
    "Jan, Feb, Mar... continuing from {prev}, next is",
    "Following the pattern Jan→Feb→Mar, after {prev} comes",
    "Last month: {prev}. This month:",
    "Next month: {next}. This month:",
    "Month-of-year sequence: {prev2}, {prev},",
    "{prev}'s successor month is",
    "{next}'s predecessor month is",
    "Going forward one month from {prev} lands on",
    "Going backward one month from {next} lands on",
    "He scheduled the trip for the month after {prev}, namely",
    "She rescheduled to the month before {next}, namely",
    "Move the meeting to the month after {prev}, that's",
    "Move the meeting to the month before {next}, that's",
    "The fiscal report is due the month right after {prev}:",
    "The audit happens the month right before {next}:",
    "Naming the month immediately after {prev}:",
    "Naming the month immediately before {next}:",
    "The annual cycle ..., {prev2}, {prev},",
    "If this month is taken to be one month past {prev}, it's",
    "If this month is taken to be one month before {next}, it's",
    "This month falls between {prev} and {next}; it's",
    "After November comes December; after December, January; after {prev},",
    "The answer to 'what comes after {prev}?' (month-wise) is",
    "The answer to 'what comes before {next}?' (month-wise) is",
    "Print the month after {prev}:",
    "Print the month before {next}:",
    "Output: month after {prev} =",
    "Output: month before {next} =",
    "If we label this month as 'm' and last month as {prev}, then m =",
    "If m-1 = {prev}, then m =",
    "If m+1 = {next}, then m =",
    "It was {prev}; the next month was",
    "It will be {next}; the month before is",
    "The twelve months are Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec; right after {prev} comes",
    "On the calendar wheel, {prev}'s neighbor (clockwise) is",
    "On the calendar wheel, {next}'s neighbor (counter-clockwise) is",
    "Month index +1 from {prev}:",
    "Month index -1 from {next}:",
    "Coming up next after {prev}:",
    "Just before {next}, we have",
    "Sandwich: {prev} on one side, on the other side is",
    "Looking at the yearly schedule: after {prev}, before {next}, sits",
    "Q: Following {prev}, what month? A:",
    "Q: Preceding {next}, what month? A:",
    "It's the start of the month after {prev}, so it's",
    "It's the end of the month before {next}, so it's",
    "From {prev}, advance one month to",
    "From {next}, retreat one month to",
    "On month {prev}+1:",
    "On month {next}-1:",
    "If {prev} just passed, this month is",
    "If {next} starts next, this month is",
    "Continuing the sequence Jan→Feb→Mar: after {prev} the next is",
    "Continuing the sequence backwards Dec→Nov→Oct: before {next} the previous is",
    "The month for which {prev} is the previous month:",
    "The month for which {next} is the next month:",
    "Sequence ..., {prev2}, {prev}, ?, {next}, ... fill in:",
    "Saying it plainly: the month after {prev} is",
    "Plainly: the month before {next} is",
    "He counted forward one month from {prev} and arrived at",
    "She counted backward one month from {next} and arrived at",
    "Spelling it out: the month after {prev} =",
    "Spelling it out: the month before {next} =",
    "By convention, {prev} is followed by",
    "By convention, {next} is preceded by",
    "The next entry in the month sequence after {prev}:",
    "The previous entry in the month sequence before {next}:",
    "Right after {prev} on the yearly cycle is",
    "Right before {next} on the yearly cycle is",
    "His anniversary falls on the month after {prev}, which is",
    "Her birthday falls on the month before {next}, namely",
    "Mark the month after {prev} on the calendar:",
    "Mark the month before {next} on the calendar:",
    "We meet the month after {prev}, that is",
    "We met the month before {next}, that is",
    "Annual planner — after {prev}, plan for",
    "Annual planner — before {next}, plan for",
    "Count one month forward from {prev}:",
    "Count one month backward from {next}:",
]

PREFIXES = [
    "",
    "Just so you know, ",
    "FYI: ",
    "Note: ",
    "Quick fact: ",
    "Remember: ",
    "By the way, ",
    "Hey, ",
    "Look, ",
    "Honestly, ",
]


def _cycle(items: list[str], idx: int, offset: int) -> str:
    return items[(idx + offset) % len(items)]


def render(template: str, items: list[str], idx: int) -> str | None:
    repl = {
        "prev": _cycle(items, idx, -1),
        "next": _cycle(items, idx, +1),
        "prev2": _cycle(items, idx, -2),
        "next2": _cycle(items, idx, +2),
        "prev3": _cycle(items, idx, -3),
        "next3": _cycle(items, idx, +3),
        "prev_long_text": "weekends",  # only used by one weekday template
    }
    try:
        return template.format(**repl)
    except KeyError:
        return None


def gen(items, stems, prefixes, family, start_id, rng):
    rows = []
    for ci, concept in enumerate(items):
        for tpl in stems:
            for pfx in prefixes:
                rendered = render(pfx + tpl, items, ci) if "{" in pfx else (pfx + (render(tpl, items, ci) or ""))
                if not rendered or "{" in rendered:
                    continue
                rows.append({
                    "id": start_id + len(rows),
                    "prompt": rendered,
                    "concept": concept,
                    "family": family,
                    "stem_idx": stems.index(tpl),
                    "prefix_idx": prefixes.index(pfx),
                })
    rng.shuffle(rows)
    for new_id, r in enumerate(rows):
        r["id"] = start_id + new_id
    return rows


def main():
    rng = random.Random(0)
    out_dir = Path("/home/rkathuria/manifolds/data")
    out_dir.mkdir(parents=True, exist_ok=True)

    weekday_rows_full = gen(WEEKDAYS, WEEKDAY_STEMS, PREFIXES, "weekday", 0, rng)
    month_rows_full = gen(MONTHS, MONTH_STEMS, PREFIXES, "month", 10**8, rng)

    # Subsample to ~280 / weekday concept and ~165 / month concept (≈ 10x the v1).
    PER_WEEKDAY = 280
    PER_MONTH = 165
    weekday_rows = []
    for c in WEEKDAYS:
        cs = [r for r in weekday_rows_full if r["concept"] == c]
        rng.shuffle(cs)
        weekday_rows.extend(cs[:PER_WEEKDAY])
    month_rows = []
    for c in MONTHS:
        cs = [r for r in month_rows_full if r["concept"] == c]
        rng.shuffle(cs)
        month_rows.extend(cs[:PER_MONTH])

    rng.shuffle(weekday_rows)
    rng.shuffle(month_rows)
    rows = weekday_rows + month_rows
    for new_id, r in enumerate(rows):
        r["id"] = new_id

    prompts_path = out_dir / "prompts_v2.jsonl"
    with open(prompts_path, "w") as f:
        for r in rows:
            f.write(json.dumps({"id": r["id"], "prompt": r["prompt"]}) + "\n")
    labels_path = out_dir / "labels_v2.jsonl"
    with open(labels_path, "w") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("id", "concept", "family", "stem_idx", "prefix_idx")}) + "\n")
    completions_path = out_dir / "completions_v2.jsonl.zst"
    cctx = zstd.ZstdCompressor(level=3)
    with open(completions_path, "wb") as f, cctx.stream_writer(f) as w:
        for r in rows:
            line = json.dumps({"prompt_id": r["id"], "completion_idx": 0,
                               "text": " " + r["concept"]}) + "\n"
            w.write(line.encode("utf-8"))

    print(f"prompts: {prompts_path}  ({len(rows)} rows)")
    print(f"  weekdays: {len(weekday_rows)}  months: {len(month_rows)}")
    counts_w = {c: sum(1 for r in weekday_rows if r['concept'] == c) for c in WEEKDAYS}
    counts_m = {c: sum(1 for r in month_rows if r['concept'] == c) for c in MONTHS}
    print("  weekday per-concept:", counts_w)
    print("  month per-concept:", counts_m)


if __name__ == "__main__":
    main()

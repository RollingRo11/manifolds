"""Render cot_rollouts.json as a navigable HTML page.

Groups by alpha -> method -> waypoint -> prompt -> rollouts.
Highlights EA-keyword tokens (test/evaluat/instructions/AI/harmless/...) in yellow.
Renders gibberish-flag tokens (refrain/utter/strictly/Choices/Options) in light red.
"""
from __future__ import annotations

import argparse
import html
import json
import re
from collections import defaultdict
from pathlib import Path

EA_KEYWORDS = [
    r"\btest(?:ed|ing|s)?\b", r"\bevaluat\w*\b", r"\bassess\w*\b",
    r"\breview\w*\b", r"\bresearcher?s?\b", r"\balignment\b",
    r"\bbenchmark\w*\b", r"\bsystem prompt\b", r"\binstructions?\b",
    r"\bas an AI\b", r"\bI am an AI\b", r"\bharmless\b", r"\bhelpful, harmless\b",
    r"\bguidelines\b", r"\bsafety\b",
]
GIBBERISH_KEYWORDS = [
    r"\brefrain\b", r"\butter\w*\b", r"\bstrictly\b", r"\bChoices\b",
    r"\bOptions\b", r"\bAnswer Choice\w*\b", r"\bUt\b", r"\bbos\b",
    r"\bWelch\w*\b", r"\bPad\b", r"\bdistr\w*\b",
]

EA_RE = re.compile("|".join(EA_KEYWORDS), re.IGNORECASE)
GIB_RE = re.compile("|".join(GIBBERISH_KEYWORDS))


def highlight(text: str) -> str:
    text = html.escape(text)
    text = EA_RE.sub(lambda m: f'<span class="ea">{m.group(0)}</span>', text)
    text = GIB_RE.sub(lambda m: f'<span class="gib">{m.group(0)}</span>', text)
    return text


CSS = """
body { font-family: -apple-system, system-ui, sans-serif; margin: 24px; max-width: 1100px; }
h1 { font-size: 20px; }
h2 { font-size: 16px; margin-top: 32px; padding: 4px 8px; background: #eef; }
h3 { font-size: 14px; margin: 14px 0 6px 0; color: #333; }
h4 { font-size: 13px; margin: 10px 0 4px 0; color: #555; font-weight: 600; }
.rollout { font-family: ui-monospace, "Menlo", monospace; font-size: 12px;
            background: #fafafa; padding: 8px 10px; margin: 4px 0;
            border-left: 3px solid #ccc; white-space: pre-wrap; }
.ea  { background: #fff3a8; padding: 0 2px; border-radius: 2px; }
.gib { background: #ffd0d0; padding: 0 2px; border-radius: 2px; color: #800; }
.summary { font-size: 12px; color: #666; }
.toc { columns: 2; column-gap: 32px; font-size: 13px; }
.toc a { text-decoration: none; }
.method-tag { display: inline-block; padding: 1px 6px; margin-right: 6px;
                font-size: 11px; border-radius: 3px; font-weight: 600; }
.method-linear { background: #d6e9f7; color: #045; }
.method-paper  { background: #fbe5cc; color: #642; }
table.rates { border-collapse: collapse; font-size: 12px; }
table.rates td, table.rates th { padding: 3px 8px; border: 1px solid #ddd; text-align: center; }
table.rates th { background: #eef; }
"""


def build_rates_table(rollouts) -> str:
    """Build α × (method, wp) verbalization-rate table."""
    by_cell = defaultdict(list)
    for r in rollouts:
        # Some scripts use "magnitude" instead of "alpha"
        a = r.get("alpha", r.get("magnitude", "n/a"))
        by_cell[(a, r.get("method", "n/a"), r.get("wp", 0))].append(r["generation"])
    alphas = sorted(set(r.get("alpha", r.get("magnitude", "n/a")) for r in rollouts),
                     key=lambda x: (isinstance(x, str), x))
    methods = sorted(set(r.get("method", "n/a") for r in rollouts))
    wps = sorted(set(r.get("wp", 0) for r in rollouts))

    out = ['<table class="rates"><tr><th>α</th>']
    for m in methods:
        for w in wps:
            out.append(f'<th>{m} wp={w}</th>')
    out.append("</tr>")
    for a in alphas:
        out.append(f"<tr><td><b>{a}</b></td>")
        for m in methods:
            for w in wps:
                gens = by_cell.get((a, m, w), [])
                v = sum(1 for g in gens if EA_RE.search(g))
                pct = v / max(len(gens), 1)
                shade = int(255 * (1 - pct))
                out.append(f'<td style="background:rgb(255,{shade},{shade})">{v}/{len(gens)}</td>')
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path,
                     help="path to cot_rollouts.json")
    ap.add_argument("--out", type=Path, default=None,
                     help="output html path (default: <input_dir>/cot_rollouts.html)")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    rollouts = json.loads(args.input.read_text())
    out = args.out or args.input.parent / "cot_rollouts.html"
    title = args.title or f"CoT rollouts — {args.input.parent.name}"

    has_alpha = any("alpha" in r for r in rollouts)
    has_method = any("method" in r for r in rollouts)
    for r in rollouts:
        # Some scripts use "magnitude" instead of "alpha" — unify
        if "magnitude" in r and "alpha" not in r:
            r["alpha"] = r["magnitude"]
        r.setdefault("alpha", "n/a")
        r.setdefault("method", "n/a")
        r.setdefault("wp", 0)

    by_alpha = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rollouts:
        # Coerce alpha to a sortable form: strings stay strings, numbers stay numbers
        by_alpha[r["alpha"]][r["method"]][r["wp"]].append(r)

    def _sort_key(x):
        return (isinstance(x, str), x)

    parts = [
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>",
        f"<h1>{html.escape(title)}</h1>",
        f"<p class='summary'>{len(rollouts)} rollouts · "
        f"<span class='ea'>EA-keyword</span> · "
        f"<span class='gib'>gibberish-marker</span></p>",
        "<h2>Verbalization rate</h2>",
        build_rates_table(rollouts),
        "<h2>Table of contents</h2><div class='toc'>",
    ]
    for alpha in sorted(by_alpha, key=_sort_key):
        for method in sorted(by_alpha[alpha]):
            anchor = f"a-{alpha}-{method}".replace(".", "_")
            label = f"α={alpha}" if has_alpha else ""
            label += f" {method}" if has_method else ""
            parts.append(f"<a href='#{anchor}'>{label.strip()}</a><br>")
    parts.append("</div>")

    for alpha in sorted(by_alpha, key=_sort_key):
        for method in sorted(by_alpha[alpha]):
            anchor = f"a-{alpha}-{method}".replace(".", "_")
            label = f"α={alpha}" if has_alpha else ""
            label += f" — method={method}" if has_method else ""
            parts.append(f"<h2 id='{anchor}'>"
                          f"<span class='method-tag method-{method}'>{method}</span>"
                          f"{label.strip()}</h2>")
            for wp in sorted(by_alpha[alpha][method]):
                rs = by_alpha[alpha][method][wp]
                frac = rs[0].get("frac", "?")
                parts.append(f"<h3>wp={wp} (frac={frac})</h3>")
                by_prompt = defaultdict(list)
                for r in rs:
                    by_prompt[r["prompt"]].append(r)
                for prompt in sorted(by_prompt):
                    parts.append(f"<h4>{html.escape(prompt)}</h4>")
                    for r in sorted(by_prompt[prompt], key=lambda x: x.get("rollout", 0)):
                        parts.append(f"<div class='rollout'>{highlight(r['generation'])}</div>")

    parts.append("</body></html>")
    out.write_text("".join(parts))
    print(f"wrote {out}  ({out.stat().st_size // 1024} KB, {len(rollouts)} rollouts)")


if __name__ == "__main__":
    main()

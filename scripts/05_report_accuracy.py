#!/usr/bin/env python3
"""Stage 3c -- the accuracy tables, grouped so only comparable things share a row.

Reads results/metrics/stage3/{summary,comparisons}.csv and writes accuracy.md.

Two tables, because there are two scopes and mixing them is the mistake this
whole design exists to avoid. The full-scope arms run on every captioned image;
the tag arms run only on the third of images ImageNet-Captions covers. A number
from one is not comparable to a number from the other, so they never appear in
the same table.

    table 1   full scope        caption vs caption_noclass, at each length
    table 2   tag subset        caption vs caption_noclass vs tags

Within a table everything shares a fold assignment, so the deltas are paired
and the p-values are the corrected resampled t-test from 04_compare_arms.py.

    python scripts/05_report_accuracy.py
    python scripts/05_report_accuracy.py --metric f1_macro --level caption
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = ROOT / "results" / "metrics" / "stage3"

SETUPS = ["w03", "w05", "w07", "w10"]
PRETTY = {"w03": "3", "w05": "5", "w07": "7", "w10": "10"}


def cell(row) -> str:
    """mean with its across-fold interval."""
    if row is None or row.empty:
        return "--"
    r = row.iloc[0]
    return f"{r['mean']:.3f} <sub>{r['ci_lo']:.3f}-{r['ci_hi']:.3f}</sub>"


def get(s: pd.DataFrame, pair: str, arm: str, setup: str):
    return s[(s.pair == pair) & (s.arm == arm) & (s.setup == setup)]


def delta(c: pd.DataFrame, pair: str, setup: str, a: str, b: str) -> str:
    """Paired difference with significance, from the corrected test."""
    m = c[(c.pair == pair) & (c.setup == setup) & (c.arm_a == a) & (c.arm_b == b)]
    if m.empty:
        return "--"
    r = m.iloc[0]
    star = "**" if r["significant"] else ""
    return f"{star}{r['delta']:+.3f}{star}"


def table_full(s, c, pairs) -> list[str]:
    out = ["| pair | words | caption | without synonyms | delta |",
           "|---|---:|---|---|---:|"]
    for pair in pairs:
        for i, setup in enumerate(SETUPS):
            out.append(
                f"| {pair if i == 0 else ''} | {PRETTY[setup]} "
                f"| {cell(get(s, pair, 'caption', setup))} "
                f"| {cell(get(s, pair, 'caption_noclass', setup))} "
                f"| {delta(c, pair, setup, 'caption', 'caption_noclass')} |")
    return out


def table_tagsub(s, c, pairs) -> list[str]:
    out = ["| pair | words | caption | without synonyms | tags | cap - tags | nosyn - tags |",
           "|---|---:|---|---|---|---:|---:|"]
    for pair in pairs:
        tags = cell(get(s, pair, "tags", "tags"))
        for i, setup in enumerate(SETUPS):
            out.append(
                f"| {pair if i == 0 else ''} | {PRETTY[setup]} "
                f"| {cell(get(s, pair, 'caption_tagsub', setup))} "
                f"| {cell(get(s, pair, 'caption_noclass_tagsub', setup))} "
                f"| {tags if i == 0 else ''} "
                f"| {delta(c, pair, setup, 'caption_tagsub', 'tags')} "
                f"| {delta(c, pair, setup, 'caption_noclass_tagsub', 'tags')} |")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", default="accuracy")
    ap.add_argument("--level", default="image", choices=["image", "caption"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    s = pd.read_csv(METRICS_DIR / "summary.csv")
    c = pd.read_csv(METRICS_DIR / "comparisons.csv")
    s = s[(s.level == args.level) & (s.metric == args.metric)]
    c = c[(c.level == args.level) & (c.metric == args.metric)]
    pairs = sorted(s.pair.unique())

    lines = [
        f"# Stage 3 -- {args.metric}, {args.level} level",
        "",
        f"Mean over 25 fits (5-fold x 5 repeats), with the across-fold 95% "
        f"interval. Folds are assigned to *images*, so the five captions of one "
        f"photograph never straddle the split, and every arm reuses the same "
        f"assignment -- which is what makes the deltas paired.",
        "",
        "`delta` columns are paired differences; **bold** is significant under "
        "the Nadeau-Bengio corrected resampled t-test with Bonferroni "
        "correction inside the comparison family. The correction is applied "
        "across all pairs and lengths in a family, so it is conservative.",
        "",
        "## Full scope -- every captioned image",
        "",
        "The synonym ablation. Both columns run on all images of the pair, so "
        "the delta is the cost of removing the class name and nothing else.",
        "",
        *table_full(s, c, pairs),
        "",
        "## Tag subset -- images ImageNet-Captions covers",
        "",
        "Descriptions against human tags. All three columns run on the same "
        "images, so the comparison is not confounded by sample. Tags have no "
        "length: they are what the uploader wrote, and the column repeats down "
        "the block.",
        "",
        *table_tagsub(s, c, pairs),
        "",
        "## Reading these",
        "",
        "- Tags are never synonym-stripped -- a tag set is what a human wrote "
        "about the photograph, and ablating it would answer a question nobody "
        "asked. So `cap - tags` compares a leaky caption to a leaky tag set, "
        "and `nosyn - tags` compares a clean caption to a leaky tag set. "
        "Neither is a like-for-like contest; both are worth reporting.",
        "- The synonym removal strips only a caption's *own* class terms, so "
        "cross-class mentions survive and become inverted indicators. "
        "See `class_leakage.md`.",
        "",
    ]

    out = Path(args.out) if args.out else METRICS_DIR / f"{args.metric}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

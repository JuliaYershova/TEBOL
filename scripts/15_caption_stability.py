#!/usr/bin/env python3
"""Stage 1 audit -- how repeatable is the captioner?

Stage 1 asked the vision model for a caption five times per image at each
length, at temperature 1.0. This measures how much those five differ. It reads
only the raw caption JSONL: no classifier, no embedding, no explainer is
involved, so what it reports is a property of the generation step alone.

Why it matters here: the explanation stability numbers elsewhere mix three
sources of randomness -- the model refit, the captioner, and LIME's sampling --
and the captioner turns out to dominate. That claim needs its own measurement
rather than being inferred by subtraction.

Per image and length, over the five repetitions:

    token Jaccard     mean pairwise overlap of the token sets. 1.0 means the
                      five captions used exactly the same words.
    content Jaccard   the same with function words removed, so the score
                      reflects what was described rather than how the sentence
                      was assembled.
    identical         share of the ten pairs that are the same string
    length sd         spread of token counts, in words
    class agreement   share of images where all five repetitions agree on
                      whether to name the class at all. Ties the caption-level
                      randomness to the leakage result: if the captioner names
                      the class in three runs out of five, the leak itself is
                      a coin flip.

    python scripts/15_caption_stability.py
    python scripts/15_caption_stability.py --pair ant_bee --limit 500
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

CAPTIONS = ROOT / "artifacts" / "captions"
TERMS = ROOT / "data" / "manifest" / "class_terms.csv"
OUT_CSV = ROOT / "results" / "metrics" / "caption_stability.csv"
OUT_MD = ROOT / "results" / "reports" / "02_caption_stability.md"

PAIRS = ("acousticguitar_violin", "ambulance_firetruck", "ant_bee",
         "cucumber_zucchini", "hotpot_vase")
MODEL = "local__qwen3-vl__32b-instruct.jsonl"

#: Removed for the content score. Deliberately a small closed list rather than
#: a linguistic resource: the point is to strip sentence scaffolding, not to
#: make a judgement about which content words matter.
STOP = {
    "a", "an", "the", "and", "or", "but", "of", "in", "on", "at", "to", "for",
    "with", "from", "by", "as", "is", "are", "was", "were", "be", "been",
    "its", "it", "this", "that", "these", "those", "there", "here", "over",
    "under", "near", "into", "onto", "up", "down", "out", "off", "against",
    "while", "during", "above", "below", "between", "through", "across",
}


def load_class_terms() -> dict[str, set[str]]:
    terms: dict[str, set[str]] = defaultdict(set)
    with TERMS.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            terms[r["class"]].add(r["term"].lower())
    return terms


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else 1.0


def pairwise(sets: list[set]) -> float:
    pairs = list(itertools.combinations(sets, 2))
    return float(np.mean([jaccard(a, b) for a, b in pairs])) if pairs else 1.0


def run_pair(pair: str, limit: int, terms: dict[str, set[str]]) -> list[dict]:
    """One row per (pair, length), aggregated over that length's images."""
    caps: dict[tuple[str, int], dict[int, dict]] = defaultdict(dict)
    with (CAPTIONS / pair / MODEL).open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("tokens"):
                caps[(r["stem"], r["n_words"])][r["rep"]] = r

    by_len: dict[int, list[dict]] = defaultdict(list)
    seen: dict[int, set] = defaultdict(set)
    for (stem, n), reps in caps.items():
        if len(reps) < 2 or (limit and len(seen[n]) >= limit and stem not in seen[n]):
            continue
        seen[n].add(stem)
        rows = list(reps.values())
        toks = [set(r["tokens"]) for r in rows]
        cont = [t - STOP for t in toks]
        strs = [r["caption_norm"] for r in rows]
        cls = rows[0]["class"]
        names = [bool(set(t) & terms.get(cls, set())) for t in toks]
        by_len[n].append({
            "jaccard": pairwise(toks),
            "jaccard_content": pairwise(cont),
            "identical": float(np.mean([a == b for a, b in
                                        itertools.combinations(strs, 2)])),
            "len_sd": float(np.std([len(r["tokens"]) for r in rows], ddof=1)),
            "len_mean": float(np.mean([len(r["tokens"]) for r in rows])),
            "class_unanimous": float(len(set(names)) == 1),
        })

    out = []
    for n, rows in sorted(by_len.items()):
        d = pd.DataFrame(rows).mean().to_dict()
        d.update(pair=pair, n_words=n, n_images=len(rows))
        out.append(d)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", action="append", choices=PAIRS)
    ap.add_argument("--limit", type=int, default=0,
                    help="images per length (0 = all)")
    args = ap.parse_args()

    terms = load_class_terms()
    rows = []
    for pair in args.pair or list(PAIRS):
        rows.extend(run_pair(pair, args.limit, terms))
        print(f"  {pair} done")

    df = pd.DataFrame(rows)[
        ["pair", "n_words", "n_images", "len_mean", "len_sd", "jaccard",
         "jaccard_content", "identical", "class_unanimous"]]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    piv = df.pivot_table(index="n_words", values=
                         ["jaccard", "jaccard_content", "identical",
                          "len_sd", "class_unanimous"])
    md = [
        "# Caption generation stability",
        "",
        "The captioner was asked five times per image at each length, "
        "temperature 1.0. This measures how much those five differ, from the "
        "raw caption files alone -- no classifier or explainer involved.",
        "",
        "| requested words | mean actual | length sd | token Jaccard "
        "| content Jaccard | identical | class agreement |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n in sorted(df.n_words.unique()):
        g = df[df.n_words == n]
        md.append(f"| {n} | {g.len_mean.mean():.1f} | {g.len_sd.mean():.2f} "
                  f"| {g.jaccard.mean():.3f} | {g.jaccard_content.mean():.3f} "
                  f"| {g.identical.mean():.3f} | {g.class_unanimous.mean():.3f} |")
    md += [
        "",
        "## What each column measures",
        "",
        "Five repetitions per image give C(5,2) = 10 pairs; every pairwise "
        "score below is the mean over those 10, then averaged over images.",
        "",
        "| column | how it is computed | range | reading |",
        "|---|---|---|---|",
        "| mean actual | token count of each caption, averaged over all "
        "captions at that length | -- | how far the model overshoots the "
        "requested word count |",
        "| length sd | standard deviation of token count across one image's 5 "
        "repetitions, averaged over images | 0+ | 0 = the same length every "
        "time; larger = the model varies how much it writes |",
        "| token Jaccard | \\|A n B\\| / \\|A u B\\| on the token sets of "
        "two repetitions | 0-1 | 1 = the five captions used exactly the same "
        "words; 0.5 = they share half |",
        "| content Jaccard | the same after removing function words (`the`, "
        "`with`, `on`, ...) | 0-1 | isolates *what was described* from *how "
        "the sentence was built*; below the token score means the agreement "
        "was partly scaffolding |",
        "| identical | share of the 10 pairs whose normalised strings match "
        "exactly | 0-1 | 1 = generation is effectively deterministic; 0 = no "
        "two runs ever coincide |",
        "| class agreement | share of images where all 5 repetitions agree on "
        "whether any class term appears | 0-1 | 1 = the class is named every "
        "time or never; below 1 = for those images, whether the caption leaks "
        "its own label is a coin flip |",
        "",
    ]
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"wrote {OUT_CSV} and {OUT_MD}")


if __name__ == "__main__":
    main()

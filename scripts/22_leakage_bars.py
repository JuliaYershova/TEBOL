#!/usr/bin/env python3
"""Stage 3n -- how often a caption names its own class, by length and by class.

`class_leakage.csv` pools every length into one number per class, which hides
the thing that matters here: the rate moves with caption length, and it moves
for two different reasons. The class *name* is one word, so a longer caption is
more likely to contain it. A *synonym* is a stylistic choice the caption model
makes when it has room, so synonym use rises faster and from a lower base.

Two charts, one per tier of data/manifest/class_terms.csv:

    tier 1   the class name and its morphological variants
             ("zucchini", "zucchinis", "hot pot", "hot-pots")
    tier 2   agreed synonyms and their variants
             ("courgette" for zucchini, "urn" for vase)

Tier 3 -- the standalone head noun of a two-word class name, "pot" for hotpot
-- is written to the CSV but not drawn: it applies to only two classes, so a
panel per class would be mostly empty.

The matching is imported from scripts/class_leakage.py rather than reimplemented,
so these bars and the ablated captions the models were trained on agree by
construction: token-level, never substring, with hyphen parts and multiword
phrases handled.

    python scripts/22_leakage_bars.py
    python scripts/22_leakage_bars.py --tiers 1 2 3
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tebol.captions_io import load_rows  # noqa: E402

CAPTIONS = ROOT / "artifacts" / "captions"
FIG = ROOT / "results" / "figures" / "leakage_bars"
OUT = ROOT / "results" / "metrics" / "class_leakage_by_tier.csv"

LENGTHS = (3, 5, 7, 10, 15, 20, 25, 30)
#: (short name for the CSV, chart title, colour)
TIERS = {1: ("Class name", "Percent of captions naming their class", "#D55E00"),
         2: ("Synonyms", "Percent of captions naming a synonym of their class",
             "#0072B2"),
         3: ("Head noun", "Percent of captions naming the head noun of their "
             "class", "#009E73")}


def leakage_module():
    """Import scripts/class_leakage.py for its term loading and matcher."""
    path = ROOT / "scripts" / "class_leakage.py"
    spec = importlib.util.spec_from_file_location("class_leakage", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def count(cl, tiers):
    """(pair, class, n_words, tier) -> share of captions containing a term."""
    # one term set per tier, so a caption can count for several independently
    per_tier = {}
    for t in tiers:
        single, phrase, _ = cl.load_terms(t)
        if t > 1:                       # load_terms is cumulative; subtract
            lo_s, lo_p, _ = cl.load_terms(t - 1)
            single = {c: single.get(c, set()) - lo_s.get(c, set()) for c in single}
            phrase = {c: [p for p in phrase.get(c, []) if p not in lo_p.get(c, [])]
                      for c in phrase}
        per_tier[t] = (single, phrase)

    files = sorted(p for p in CAPTIONS.glob("*/*.jsonl")
                   if not p.stem.endswith(cl.NOCLASS))
    if not files:
        sys.exit(f"no caption files under {CAPTIONS}")

    n: dict = defaultdict(Counter)
    for path in files:
        pair = path.parent.name
        for r in load_rows(path):
            toks, cls, w = r.get("tokens") or [], r["class"], r["n_words"]
            k = (pair, cls, w)
            n[k]["captions"] += 1
            for t, (single, phrase) in per_tier.items():
                if cl.find(toks, cls, single, phrase):
                    n[k][t] += 1
        print(f"  read {pair}")

    rows = []
    for (pair, cls, w), c in sorted(n.items()):
        for t in tiers:
            rows.append({"pair": pair, "class": cls, "words": w, "tier": t,
                         "tier_name": TIERS[t][0], "captions": c["captions"],
                         "with_term": c[t],
                         "pct": 100 * c[t] / c["captions"] if c["captions"] else 0.0})
    return pd.DataFrame(rows)


def chart(df, tier, out):
    import matplotlib.pyplot as plt

    _, title, color = TIERS[tier]
    d = df[(df.tier == tier) & df.words.isin(LENGTHS)]
    classes = list(dict.fromkeys(
        df.sort_values(["pair", "class"])["class"]))          # pair-mates adjacent
    ncol = 5
    nrow = -(-len(classes) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.55 * ncol, 2.35 * nrow),
                             sharex=True, sharey=True, squeeze=False)

    for k, cls in enumerate(classes):
        ax = axes[k // ncol][k % ncol]
        g = d[d["class"] == cls].set_index("words").reindex(LENGTHS)
        pct = g["pct"].fillna(0).to_numpy()
        ax.bar(range(len(LENGTHS)), pct, color=color, width=0.72)
        for i, v in enumerate(pct):
            ax.text(i, v + 1.5, f"{v:.0f}", ha="center", fontsize=6.5, color="#444")
        ax.set_title(cls, fontsize=9)
        ax.set_xticks(range(len(LENGTHS)))
        ax.set_xticklabels(LENGTHS, fontsize=7.5)
        ax.tick_params(axis="y", labelsize=7.5)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for k in range(len(classes), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    axes[0][0].set_ylim(0, 105)
    for r in range(nrow):
        axes[r][0].set_ylabel("% of captions", fontsize=8.5)
    for c in range(ncol):
        axes[nrow - 1][c].set_xlabel("Caption length (words)", fontsize=8.5)
    fig.suptitle(title, fontsize=11.5, y=1.0)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{out}.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote {FIG / f'{out}.png'}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiers", nargs="*", type=int, default=[1, 2, 3],
                    choices=[1, 2, 3], help="tiers to count (1 and 2 are drawn)")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")

    cl = leakage_module()
    print("counting terms in the captions:")
    df = count(cl, sorted(set(args.tiers)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}")

    for tier, stem in ((1, "class_name"), (2, "synonyms")):
        if tier in args.tiers:
            chart(df, tier, stem)

    print("\n% of captions containing the term, by length:\n")
    for tier in sorted(set(args.tiers)):
        t = df[df.tier == tier].pivot_table(index="class", columns="words",
                                            values="pct")
        print(f"--- tier {tier}: {TIERS[tier][1]} ---")
        print(t[list(LENGTHS)].round(1).to_string(), "\n")


if __name__ == "__main__":
    main()

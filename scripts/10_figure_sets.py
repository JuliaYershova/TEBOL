#!/usr/bin/env python3
"""Stage 3h -- the AOPC figures, organised into the six comparisons.

    01_full_captions/              SMER vs LIME, captions as generated
    02_without_synonyms/           SMER vs LIME, class terms removed
    03_tags/                       SMER vs LIME, ImageNet-Captions tags
    04_full_vs_without_synonyms/   both representations, both methods
    05_full_captions_vs_tags/      captions vs tags, both methods
    06_all_to_all/                 every representation, both methods

01-03 ask "do the two explainers agree on this representation"; 04-06 ask "does
the representation matter", carrying both explainers so the two questions can be
read off one figure without conflating them.

**Colour means representation, line style means method.** Captions are blue,
synonyms-removed vermillion, tags green; SMER is solid with a filled marker,
LIME dashed with an open one. A reader who learns that once can read all six
folders, and in 06 the six lines stay separable.

**Scope in 05 and 06.** Tags exist for about a third of the images, so any
figure containing a tag line uses the tag-covered subset for *every* line --
`caption_tagsub` rather than `caption`. Putting full-corpus captions beside
tag-subset tags would compare representations across different photographs and
confound the two. 01-04 use the full corpus, since no tag line appears.

k starts at 1: AOPC is zero at k=0 by construction, so the point carries no
information and forces the y-axis to zero, compressing everything above it.

    python scripts/10_figure_sets.py
    python scripts/10_figure_sets.py --pair hotpot_vase
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

AOPC_CSV = ROOT / "results" / "metrics" / "aopc.csv"
OUT = ROOT / "results" / "figures" / "aopc"

X_LABEL = "Number of words removed"
Y_LABEL = "Probability decrease"
K_MIN = 1

SMER_RANKING = "smer_subsample"
LIME_RANKING = "lime_bow"          # LIME's own defaults; see 09_lime.py

#: arm -> (legend label, colour, marker). Colour identifies the arm; SMER is
#: solid with a filled marker, LIME dashed with a hollow one, so method never
#: competes with representation for the same visual channel.
#:
#: `caption` and `caption_tagsub` no longer share a colour. They did while no
#: figure carried both, but 06 now shows the full corpus beside the tag-covered
#: subset, and two lines meaning different image sets cannot look identical.
#: Shade-within-a-family was tried first -- dark blue for the corpus, light
#: blue for the subset -- and the two blues were not separable on the page.
#: Each arm now takes its own hue from the Okabe-Ito set instead. Markers are
#: chosen for silhouette rather than family: circle, square, triangle, diamond,
#: star. Up- and down-triangles were tried first and read as the same shape at
#: print size; a cross read as a plotting artefact rather than a marker. The
#: diamond is a rotated square, but corner-up against flat-top separates them
#: at this size.
STYLE = {
    "caption": ("Captions", "#0072B2", "o"),
    "caption_noclass": ("Captions, synonyms removed", "#D55E00", "s"),
    "tags": ("ImageNet tags", "#009E73", "^"),
    "caption_tagsub": ("Captions, tag subset", "#CC79A7", "D"),
    "caption_noclass_tagsub":
        ("Captions, synonyms removed, tag subset", "#E69F00", "*"),
}

WORDS = {f"w{n:02d}": f"{n}words" for n in (3, 5, 7, 10, 15, 20, 25, 30)}

#: folder -> (arms, needs a caption length, note appended to the labels)
SETS = {
    "01_full_captions": (["caption"], True, ""),
    "02_without_synonyms": (["caption_noclass"], True, ""),
    "03_tags": (["tags"], False, ""),
    "04_full_vs_without_synonyms": (["caption", "caption_noclass"], True, ""),
    "05_full_captions_vs_tags": (["caption_tagsub", "tags"], True, ""),
    # Deliberately mixes scopes: the first two lines are the full corpus, the
    # last two the third of images ImageNet-Captions covers. That is what makes
    # it all-to-all, and it is why `caption` and `caption_tagsub` are both here
    # and separately coloured -- the pair of them prices the restriction, so a
    # reader can tell a representation effect from a sample effect.
    "06_all_to_all":
        (["caption", "caption_noclass", "tags", "caption_tagsub"], True, ""),
}


def series(df, pair, arm, setup, ranking):
    want = "tags" if arm == "tags" else setup
    g = df[(df.pair == pair) & (df.arm == arm) & (df.setup == want)
           & (df.ranking == ranking)]
    return g[(g.k >= K_MIN)].sort_values("k")


def figure(df, pair, arms, setup, note, path, plt) -> bool:
    lines = []
    for arm in arms:
        s, l = (series(df, pair, arm, setup, SMER_RANKING),
                series(df, pair, arm, setup, LIME_RANKING))
        if s.empty or l.empty:
            continue
        lines.append((arm, s, l))
    if not lines:
        return False

    # Each line runs to its own k. Truncating everything to the shortest was
    # wrong: tags hold ~7 words and cap at k=5, so a 10-word caption figure lost
    # four of its nine points to a line that had simply run out of words. A line
    # that stops early is self-evident on the page; a caption curve silently cut
    # at someone else's limit is not.
    k_max = max(int(s["k"].max()) for _, s, _ in lines)
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    lo, hi = np.inf, -np.inf
    single = len(arms) == 1

    for arm, s, l in lines:
        label, color, mk = STYLE[arm]
        for g, meth, ls, fill in (
                (s, "SMER", "-", color), (l, "LIME", (0, (4, 2)), "none")):
            name = meth if single else f"{label} ({meth})"
            ax.plot(g["k"], g["mean"], color=color, linestyle=ls, marker=mk,
                    markersize=4.4, markerfacecolor=fill, markeredgecolor=color,
                    markeredgewidth=1.0, linewidth=1.1, label=name)
            ax.fill_between(g["k"], g["ci_lo"], g["ci_hi"], color=color,
                            alpha=0.13, linewidth=0)
            lo = min(lo, float(g["ci_lo"].min()))
            hi = max(hi, float(g["ci_hi"].max()))

    pad = (hi - lo) * 0.08
    ax.set_xlabel(X_LABEL, fontsize=10)
    ax.set_ylabel(Y_LABEL, fontsize=10)
    # Every integer is a tick up to ~12 removals; beyond that they collide
    # (w30 runs to k=29 and the labels ran together into "101112131415").
    stride = 1 if k_max <= 12 else (2 if k_max <= 20 else 3)
    ticks = list(range(K_MIN, k_max + 1, stride))
    if ticks[-1] != k_max:
        ticks.append(k_max)
    ax.set_xticks(ticks)
    ax.set_ylim(max(0.0, lo - pad), hi + pad)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=7 if len(lines) > 2 else 8, frameon=False,
              loc="upper left", handlelength=2.4)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", action="append")
    ap.add_argument("--keep-old", action="store_true",
                    help="do not clear previously generated figures first")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(AOPC_CSV)
    pairs = args.pair or sorted(df["pair"].unique())

    if OUT.exists() and not args.keep_old:
        shutil.rmtree(OUT)

    total = 0
    for folder, (arms, per_length, note) in SETS.items():
        n = 0
        for pair in pairs:
            if per_length:
                for setup, words in WORDS.items():
                    if figure(df, pair, arms, setup, note,
                              OUT / folder / f"{pair}__{words}", plt):
                        n += 1
            else:
                if figure(df, pair, arms, "tags", note,
                          OUT / folder / f"{pair}", plt):
                    n += 1
        print(f"  {folder:<32} {n:3d} figures")
        total += n
    print(f"\nwrote {total} figures (png + pdf) under {OUT}")


if __name__ == "__main__":
    main()

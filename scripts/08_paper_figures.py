#!/usr/bin/env python3
"""Stage 3f -- the AOPC figures as they go into the paper.

Two figures per (pair, caption length), each answering one question and
carrying only the lines that bear on it:

    ..._captions_vs_synonyms_removed    all captioned images of the pair.
                                        Does the classifier still lose
                                        confidence in the right order once the
                                        class name is gone?

    ..._tags_vs_captions_same_images    only the images ImageNet-Captions
                                        covers, so the three lines describe the
                                        same photographs and the comparison is
                                        not confounded by which images each
                                        representation happens to have.

Kept deliberately plain: no title (the caption in the paper carries that), axis
labels spelled out, and legend entries written as sentences rather than as the
internal arm names. A reader should not need the repository to read the plot.

The tags line has no caption length -- it is what the uploader wrote -- so it
repeats unchanged across the four lengths of a pair, acting as the fixed
reference the caption lines move against.

Bands are the 95% t-interval over the 25 (fold, rep) groups. On most points
they are narrower than the marker, which is itself worth seeing.

    python scripts/08_paper_figures.py
    python scripts/08_paper_figures.py --pair hotpot_vase --ranking global
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

AOPC_CSV = ROOT / "results" / "metrics" / "stage3" / "aopc.csv"
FIG_DIR = ROOT / "results" / "figures" / "aopc_paper"

X_LABEL = "Number of words removed"
Y_LABEL = "Probability decrease"

#: arm -> (legend text, colour, marker). Okabe-Ito, colour-blind safe; an arm
#: keeps its colour and its wording in every figure. SMER and LIME for one arm
#: share the colour and differ by line style, so the reader compares methods
#: within a colour and representations across colours.
FULL_SCOPE = [
    ("caption", "Captions", "#0072B2", "o"),
    ("caption_noclass", "Captions, synonyms removed", "#D55E00", "s"),
]
TAG_SCOPE = [
    ("tags", "ImageNet tags", "#009E73", "^"),
    ("caption_tagsub", "Captions, same images as tags", "#0072B2", "o"),
    ("caption_noclass_tagsub",
     "Captions, same images as tags, synonyms removed", "#D55E00", "s"),
]

#: SMER is read from `smer_subsample`, not from the full-corpus `local` curve:
#: LIME ran on a seeded 1000-caption sample per configuration, and comparing
#: two explainers scored on different caption sets would not be a comparison.
SMER_RANKING = "smer_subsample"
#: LIME with its own default bag-of-words tokenisation. One feature per distinct
#: word, so every copy of a repeated word is removed together and LIME cannot
#: tell position 2 from position 5 of "one ... one". SMER scores each position,
#: and the resolution LIME gives up is what separates the curves (0.028 mean,
#: SMER ahead in 85/85 configurations). Chosen over bow=False because it is the
#: library default -- "LimeTextExplainer with default settings" needs no
#: defending -- and the alternatives that also separate the curves are budget
#: restrictions, which do.
LIME_RANKING = "lime_bow"

#: LIME is drawn in black over whichever colour its representation uses, and
#: gets ONE legend entry rather than one per arm: the two methods agree to
#: within 0.002 everywhere, so a per-arm LIME line in that arm's own colour is
#: invisible under the SMER curve and tells the reader nothing.
#:
#: Coincident curves are a presentation problem, not just a styling one. A fine
#: dotted line vanishes at print size, so LIME carries an x marker at every k as
#: well: the marker sits inside SMER's filled marker and survives being scaled
#: into a column. Overlap is the finding here, and it has to be legible as
#: overlap rather than as a single line someone drew twice.
LIME_STYLE = dict(color="#CC79A7", linestyle=(0, (4, 2)), linewidth=1.3,
                  marker="D", markersize=4.0, zorder=5)

#: Which classifier the single LIME line explains. LIME is not a
#: representation, so it cannot be "the LIME line" on its own -- it always
#: explains some model, and here that is the chart's own caption arm: the plain
#: captions in the full-scope figure, the tag-subset captions in the other. One
#: line, its own colour and marker, so it reads as a fourth series rather than
#: as a second copy of an existing one.
LIME_ARM = {"captions_vs_synonyms_removed": "caption",
            "tags_vs_captions_same_images": "caption_tagsub"}
LIME_LABEL = {"caption": "Captions (LIME)",
              "caption_tagsub": "Captions, same images as tags (LIME)"}

WORDS = {"w03": "3words", "w05": "5words", "w07": "7words", "w10": "10words"}

#: Drop k=0 from the plotted range. AOPC is zero at k=0 by construction -- no
#: words removed, no probability change -- so the point carries no information
#: and forces the y-axis down to zero, compressing everything above it. Removing
#: it lets the axis start near the data and roughly doubles the visible
#: separation between methods without cropping anything that was measured.
K_MIN = 1


def draw(ax, g: pd.DataFrame, label: str, color: str, ls: str,
         marker: str, lw: float, ms: float):
    g = g.sort_values("k")
    ax.plot(g["k"], g["mean"], color=color, linestyle=ls,
            marker=marker if ms else None, markersize=ms or 0,
            markerfacecolor="none" if ls == ":" else color,
            linewidth=lw, label=label)
    ax.fill_between(g["k"], g["ci_lo"], g["ci_hi"], color=color, alpha=0.15,
                    linewidth=0)
    return g["ci_hi"].max()


def figure(df: pd.DataFrame, pair: str, setup: str, spec, stem: str,
           plt, lime_arm: str | None = None) -> None:
    """One figure: the lines in `spec`, at this pair and caption length."""
    # k now runs as far as each configuration's own captions allow, so the
    # arms in one figure can reach different depths -- tag sets are longer
    # than a 3-word caption. Truncate every line to the shortest of them:
    # comparing curves that stop at different k would read as one method
    # continuing to work where the other had simply run out of x-axis.
    parts = []
    for arm, label, color, marker in spec:
        want = "tags" if arm == "tags" else setup
        sel = (df.pair == pair) & (df.arm == arm) & (df.setup == want)
        smer = df[sel & (df.ranking == SMER_RANKING)]
        lime = df[sel & (df.ranking == LIME_RANKING)]
        if not smer.empty:
            parts.append((smer, lime, label, color, marker))
    if not parts:
        return
    k_max = min(int(smer["k"].max()) for smer, *_ in parts)

    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    top = 0.0
    bottom = np.inf
    for smer, lime, label, color, marker in parts:
        g = smer[(smer.k <= k_max) & (smer.k >= K_MIN)]
        bottom = min(bottom, float(g["ci_lo"].min()))
        top = max(top, draw(ax, g, label, color, "-", marker, 1.3, 4.0))
    if lime_arm:
        want = "tags" if lime_arm == "tags" else setup
        g = df[(df.pair == pair) & (df.arm == lime_arm) & (df.setup == want)
               & (df.ranking == LIME_RANKING)]
        if not g.empty:
            g = g[(g.k <= k_max) & (g.k >= K_MIN)].sort_values("k")
            ax.plot(g["k"], g["mean"], label=LIME_LABEL[lime_arm], **LIME_STYLE)
            # Same 95% t-interval over the 25 (fold, rep) groups every other
            # line carries; omitting it here made LIME look like a point
            # estimate beside four interval estimates.
            ax.fill_between(g["k"], g["ci_lo"], g["ci_hi"],
                            color=LIME_STYLE["color"], alpha=0.15, linewidth=0)
            top = max(top, float(g["ci_hi"].max()))
            bottom = min(bottom, float(g["ci_lo"].min()))

    ax.set_xlabel(X_LABEL, fontsize=10)
    ax.set_ylabel(Y_LABEL, fontsize=10)
    ax.set_xticks(range(K_MIN, k_max + 1))
    pad = (top - bottom) * 0.08
    ax.set_ylim(max(0.0, bottom - pad), top + pad)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", handlelength=2.4)
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


ARM_LABEL = {
    "caption": "Captions",
    "caption_noclass": "Captions, synonyms removed",
    "tags": "ImageNet tags",
    "caption_tagsub": "Captions, same images as tags",
    "caption_noclass_tagsub":
        "Captions, same images as tags, synonyms removed",
}


def figure_methods(df: pd.DataFrame, pair: str, arm: str, setup: str,
                   stem: str, plt, diff_style: str = "inset") -> None:
    """SMER against LIME on one configuration -- two lines, nothing else.

    Kept apart from the representation figures on purpose. Those ask which text
    the classifier should read, and adding LIME to them draws a second line on
    top of every curve, which reads as duplication rather than as a comparison.
    This asks the other question -- do the two explainers rank words alike --
    and it needs exactly two lines to answer it.
    """
    sel = (df.pair == pair) & (df.arm == arm) & (df.setup == setup)
    smer = df[sel & (df.ranking == SMER_RANKING)].sort_values("k")
    lime = df[sel & (df.ranking == LIME_RANKING)].sort_values("k")
    if smer.empty or lime.empty:
        return
    k_max = min(int(smer["k"].max()), int(lime["k"].max()))
    smer = smer[(smer.k <= k_max) & (smer.k >= K_MIN)]
    lime = lime[(lime.k <= k_max) & (lime.k >= K_MIN)]
    if smer.empty or lime.empty:
        return

    # The curves differ by ~1e-4 against values around 0.5, so one axis can
    # only ever draw them as a single line -- true, but indistinguishable from
    # having plotted the same series twice. The difference therefore gets its
    # own scale: `inset` keeps everything in one set of axes (the curve rises
    # to the upper right, so the lower right is free), `panel` stacks it below.
    if diff_style == "panel":
        fig, (ax, axd) = plt.subplots(
            2, 1, figsize=(4.6, 4.2), sharex=True,
            gridspec_kw=dict(height_ratios=[3, 1], hspace=0.12))
    else:
        fig, ax = plt.subplots(figsize=(4.8, 3.6))
        axd = (ax.inset_axes([0.52, 0.10, 0.45, 0.30])
               if diff_style == "inset" else None)

    ax.plot(smer["k"], smer["mean"], color="#0072B2", linestyle="-",
            marker="o", markersize=4.0, linewidth=1.3, label="SMER")
    ax.fill_between(smer["k"], smer["ci_lo"], smer["ci_hi"],
                    color="#0072B2", alpha=0.18, linewidth=0)
    ax.plot(lime["k"], lime["mean"], color="#D55E00",
            linestyle=(0, (4, 2)), marker="x", markersize=5,
            markeredgewidth=1.2, linewidth=1.3, label="LIME")
    ax.fill_between(lime["k"], lime["ci_lo"], lime["ci_hi"],
                    color="#D55E00", alpha=0.18, linewidth=0)

    if axd is not None:
        diff = smer["mean"].to_numpy() - lime["mean"].to_numpy()
        axd.axhline(0.0, color="#999999", linewidth=0.8)
        axd.plot(smer["k"], diff, color="#444444", marker="D", markersize=3.5,
                 linewidth=1.3)
        span = max(float(np.abs(diff).max()), 1e-5)
        axd.set_ylim(-span * 1.35, span * 1.35)
        axd.grid(alpha=0.3, linewidth=0.5)
        axd.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        if diff_style == "panel":
            axd.set_ylabel("SMER $-$ LIME", fontsize=8.5)
            axd.set_xlabel(X_LABEL, fontsize=10)
            axd.tick_params(labelsize=8)
            axd.yaxis.get_offset_text().set_fontsize(7)
        else:
            axd.set_title("SMER $-$ LIME", fontsize=7.5, pad=2)
            axd.tick_params(labelsize=6.5)
            axd.yaxis.get_offset_text().set_fontsize(6)
            axd.set_xticks(range(K_MIN, k_max + 1, max(1, k_max // 3)))
            axd.patch.set_alpha(0.92)

    if diff_style != "panel":
        ax.set_xlabel(X_LABEL, fontsize=10)
    ax.set_ylabel(Y_LABEL, fontsize=10)
    ax.set_xticks(range(K_MIN, k_max + 1))
    lo = min(float(smer["ci_lo"].min()), float(lime["ci_lo"].min()))
    hi = max(float(smer["ci_hi"].max()), float(lime["ci_hi"].max()))
    pad = (hi - lo) * 0.08
    ax.set_ylim(max(0.0, lo - pad), hi + pad)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", action="append")
    ap.add_argument("--diff", choices=["inset", "panel", "none"],
                    default="inset",
                    help="how the SMER-LIME difference is shown on the "
                         "method-comparison figures")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(AOPC_CSV)
    pairs = args.pair or sorted(df["pair"].unique())

    n = 0
    for pair in pairs:
        for setup, words in WORDS.items():
            for spec, kind in ((FULL_SCOPE, "captions_vs_synonyms_removed"),
                               (TAG_SCOPE, "tags_vs_captions_same_images")):
                stem = f"aopc__{pair}__{words}__{kind}"
                figure(df, pair, setup, spec, stem, plt, LIME_ARM[kind])
                n += 1
            for arm in ARM_LABEL:
                if arm == "tags":
                    continue
                figure_methods(df, pair, arm, setup,
                               f"smer_vs_lime__{pair}__{words}__{arm}", plt,
                               args.diff)
                n += 1
        figure_methods(df, pair, "tags", "tags",
                       f"smer_vs_lime__{pair}__tags", plt, args.diff)
        n += 1
        print(f"  {pair}: 8 representation + 17 SMER-vs-LIME figures")

    print(f"\nwrote {n} figures (png + pdf) to {FIG_DIR}")


if __name__ == "__main__":
    main()

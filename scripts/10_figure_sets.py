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
LIME_RANKING = "lime"          # bow=False: one feature per position.

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
    "gpt4o": ("GPT-4o descriptions", "#0072B2", "o"),
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
    # 06 puts tags on the same axes as descriptions, which compares a list of
    # 7-9 tags against ~5.7 word tokens over different image sets. 07 keeps the
    # three description conditions, which are mutually comparable; set 05 shows
    # tags against the description arm restricted to the images tags cover.
    "07_captions_only":
        (["caption", "caption_noclass", "caption_tagsub"], True, ""),
    # Tags alone: they are a different kind of input (7-9 items against ~5.7
    # word tokens, on a third of the images), so they get their own axes.
    "08_tags_only":
        (["tags"], False, ""),
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


def grid_figure(df, pairs, arms, setup, path, plt, lime_ranking, ncols, fs,
                smer_only=False, xlabel=None, overlay=False):
    """Every pair as a panel of one figure, with a single shared legend.

    overlay: for curves that coincide. SMER is drawn as a thick line with blue
    dots, LIME as a thin dashed dark line with small hollow blue dots on top, so
    both stay visible where the curves overlap.
    """
    nrow = -(-len(pairs) // ncols)
    fig, axes = plt.subplots(nrow, ncols, figsize=(2.5 * ncols, 2.3 * nrow),
                             squeeze=False)
    flat = axes.ravel()
    for ax in flat[len(pairs):]:
        ax.axis("off")
    handles, labels = [], []
    single = len(arms) == 1
    for ax, pair in zip(flat, pairs):
        lines = []
        for arm in arms:
            a = series(df, pair, arm, setup, SMER_RANKING)
            b = series(df, pair, arm, setup, lime_ranking)
            if not a.empty and not b.empty:
                lines.append((arm, a, b))
        k_max = max(int(a["k"].max()) for _, a, _ in lines)
        lo, hi = np.inf, -np.inf
        for arm, a, b in lines:
            label, color, mk = STYLE[arm]
            # Same styling as the per-pair figures: solid line and filled
            # marker for SMER, dashed line and hollow marker for LIME, one
            # colour per arm, a CI band on each.
            draw = [(a, "SMER", "-", color)]
            if not smer_only:
                draw.append((b, "LIME", (0, (4, 2)), "none"))
            for g, meth, ls, fill in draw:
                name = (label if smer_only else
                        (meth if single else f"{label} ({meth})"))
                if overlay and meth == "LIME":
                    ln, = ax.plot(g["k"], g["mean"], color="#222222",
                                  linestyle=(0, (3, 2)), marker=mk,
                                  markersize=3.2, markerfacecolor="white",
                                  markeredgecolor=color, markeredgewidth=0.9,
                                  linewidth=0.9, label=name, zorder=4)
                elif overlay:
                    ln, = ax.plot(g["k"], g["mean"], color=color, linestyle=ls,
                                  marker=mk, markersize=5.0, markerfacecolor=fill,
                                  markeredgecolor=color, markeredgewidth=1.0,
                                  linewidth=2.6, label=name, zorder=3)
                else:
                    ln, = ax.plot(g["k"], g["mean"], color=color, linestyle=ls,
                                  marker=mk, markersize=4.0, markerfacecolor=fill,
                                  markeredgecolor=color, markeredgewidth=1.0,
                                  linewidth=1.1, label=name)
                ax.fill_between(g["k"], g["ci_lo"], g["ci_hi"], color=color,
                                alpha=0.13, linewidth=0)
                if name not in labels:
                    labels.append(name); handles.append(ln)
                lo = min(lo, float(g["ci_lo"].min()))
                hi = max(hi, float(g["ci_hi"].max()))
        pad = (hi - lo) * 0.08
        ax.set_title(pair.replace("_", " / "), fontsize=9.5 * fs)
        ax.set_xlabel(xlabel or X_LABEL, fontsize=8.5 * fs)
        ax.set_xticks(list(range(K_MIN, k_max + 1)))
        ax.set_ylim(max(0.0, lo - pad), hi + pad)
        ax.grid(alpha=0.3, linewidth=0.5)
        ax.tick_params(labelsize=7.5 * fs)
    for r in range(nrow):
        flat[r * ncols].set_ylabel(Y_LABEL, fontsize=9 * fs)
    fig.legend(handles, labels, loc="lower center",
               ncol=min(4, len(labels)), fontsize=8 * fs, frameon=False,
               bbox_to_anchor=(0.5, -0.10))
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}.pdf")


#: LIME perturbation budgets that were run across lengths 3-10, newest first.
BUDGETS = (("lime", 5000), ("lime_ns500", 500), ("lime_ns100", 100),
           ("lime_ns50", 50))


def budget_figure(df, pairs, arm, path, plt, fs):
    """AOPC deficit against SMER as a function of LIME's perturbation budget.

    SMER is exact: one dot product per word, no sampling. LIME approaches the
    same ranking only as its sample count grows, and needs more samples as the
    description gets longer. Each point is the trapezoidal AOPC of SMER minus
    that of LIME, averaged over the five tasks, with a 95% t-interval.
    """
    from scipy import stats
    lengths = [3, 5, 7, 10]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for (rank, n_samples), mk in zip(BUDGETS, ("o", "s", "^", "D")):
        xs, mus, los, his = [], [], [], []
        for L in lengths:
            deficits = []
            for pair in pairs:
                a = series(df, pair, arm, f"w{L:02d}", SMER_RANKING)
                b = series(df, pair, arm, f"w{L:02d}", rank)
                if a.empty or b.empty:
                    continue
                deficits.append(float(np.trapezoid(a["mean"], a["k"])
                                      - np.trapezoid(b["mean"], b["k"])))
            if not deficits:
                continue
            d = np.asarray(deficits)
            h = (stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
                 if len(d) > 1 else 0.0)
            xs.append(L); mus.append(d.mean())
            los.append(d.mean() - h); his.append(d.mean() + h)
        ax.plot(xs, mus, marker=mk, markersize=4.5, linewidth=1.3,
                label=f"{n_samples:,} samples")
        ax.fill_between(xs, los, his, alpha=0.15, linewidth=0)
    ax.axhline(0, color="0.4", linewidth=0.8, linestyle=":")
    ax.set_xticks(lengths)
    ax.set_xlabel("Requested description length (words)", fontsize=9 * fs)
    ax.set_ylabel("AOPC deficit against SMER", fontsize=9 * fs)
    ax.tick_params(labelsize=8 * fs)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.legend(title="LIME budget", fontsize=8 * fs, title_fontsize=8 * fs,
              frameon=False, loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}.pdf")


#: The GPT-4o configuration's AOPC, lifted from the Diplom notebooks into
#: results/metrics/aopc_gpt.csv. One 80/20 split, so there is no interval to
#: draw, and the ranking is the global top-20 word list rather than a
#: per-description one, so these curves are not interchangeable with the
#: Qwen3-VL ones -- they are reported as the earlier configuration's result.
GPT_CSV = ROOT / "results" / "metrics" / "aopc_gpt.csv"


def gpt_figure(df, pairs, path, plt, ncols, fs, xlabel=None):
    """The GPT-4o configuration, reproducing its published figure exactly.

    Deliberately NOT styled like the Qwen grids: this is the earlier
    configuration's published result, so it keeps its own panel order, layout,
    per-panel axis ranges, two-colour scheme and end-of-line labels.
    """
    from matplotlib import gridspec

    BLUE, ORANGE = "#4472C4", "#ED7D31"
    # (pair, sub-label, y max, y tick step, label order top-first). The order
    # is the published one; where the two curves end within a hair of each
    # other the labels are pushed apart so both stay readable.
    PANELS = [
        ("hotpot_vase",           "(hotpot, vase)",            0.025, 0.005, ("SMER", "LIME")),
        ("cucumber_zucchini",     "(zucchini, cucumber)",      0.35,  0.05,  ("SMER", "LIME")),
        ("acousticguitar_violin", "(acoustic guitar, violin)", 0.35,  0.05,  ("SMER", "LIME")),
        ("ant_bee",               "(ant, bee)",                0.10,  0.02,  ("LIME", "SMER")),
        ("ambulance_firetruck",   "(ambulance, fire truck)",   0.20,  0.05,  ("SMER", "LIME")),
    ]
    letters = "abcde"

    fig = plt.figure(figsize=(9.6, 11.0))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=1.05)
    slots = [gs[0, 0:2], gs[0, 2:4], gs[1, 0:2], gs[1, 2:4], gs[2, 1:3]]

    for (pair, sub, ymax, step, order), slot, letter in zip(PANELS, slots, letters):
        ax = fig.add_subplot(slot)
        ends = {}
        for rank, colour, name in (("smer", BLUE, "SMER"),
                                   ("lime", ORANGE, "LIME")):
            g = df[(df.pair == pair) & (df.ranking == rank)].sort_values("k")
            if g.empty:
                continue
            xs = [0] + [int(v) for v in g["k"]]
            ys = [0.0] + [float(v) for v in g["mean"]]
            ax.plot(xs, ys, color=colour, linewidth=1.8, marker="o",
                    markersize=4.0, markerfacecolor=colour,
                    markeredgecolor=colour, clip_on=False, zorder=3)
            ends[name] = (xs[-1], ys[-1], colour)

        ax.set_ylim(0, ymax)
        ticks = np.arange(0, ymax + step / 2, step)
        ax.set_yticks(ticks)
        dec = 3 if step < 0.01 else 2
        ax.set_yticklabels([f"{t:.{dec}f}" for t in ticks])
        ax.set_xlim(-0.15, 6.15)
        ax.set_xticks(range(0, 7))
        ax.set_xlabel("Number of words removed", fontsize=11)
        ax.set_ylabel("Probability decrease", fontsize=11)
        ax.tick_params(labelsize=10, length=0)
        ax.yaxis.grid(True, color="#D9D9D9", linewidth=0.9)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#BFBFBF")

        # Labels at the end of each line, in the published order, separated
        # when the two curves finish too close to label independently.
        if len(ends) == 2:
            top, bot = order
            yt, yb = ends[top][1], ends[bot][1]
            gap = 0.075 * ymax
            if abs(yt - yb) < gap:
                mid = (yt + yb) / 2
                yt, yb = mid + gap / 2, mid - gap / 2
            for name, y in ((top, yt), (bot, yb)):
                ax.annotate(name, xy=(6, y), xytext=(7, 0),
                            textcoords="offset points", color=ends[name][2],
                            fontsize=10.5, fontweight="bold",
                            va="center", ha="left", annotation_clip=False)
        else:
            for name, (x, y, colour) in ends.items():
                ax.annotate(name, xy=(x, y), xytext=(7, 0),
                            textcoords="offset points", color=colour,
                            fontsize=10.5, fontweight="bold",
                            va="center", ha="left", annotation_clip=False)

        ax.annotate(f"({letter})", xy=(0.5, -0.30), xycoords="axes fraction",
                    ha="right", va="top", fontsize=11)
        ax.annotate(f" {sub}", xy=(0.5, -0.30), xycoords="axes fraction",
                    ha="left", va="top", fontsize=11, fontstyle="italic")

    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}.pdf")


def gpt_figure_qwen(df, pairs, path, plt, ncols, fs, xlabel=None):
    """The GPT-4o values, styled exactly like the Qwen3-VL grids.

    Same colour, markers, line styles, fonts, panel layout and shared bottom
    legend as grid_figure, so the two configurations can be shown as one set.
    One 80/20 split, so no interval band is drawn. The published layout is
    gpt_figure; this is the alternative presentation.

    Three things are taken from the published figure rather than from
    grid_figure, because without them the same numbers read as a different
    result: k = 0 is plotted (the rise from zero is most of the signal here)
    and the y-axis starts at zero (these curves are flat, and autoscaling a
    flat curve turns a 0.002 difference into a full-height swing). Panel order
    is the Qwen grids', so the figures can be read side by side.
    """
    nrow = -(-len(pairs) // ncols)
    fig, axes = plt.subplots(nrow, ncols, figsize=(2.5 * ncols, 2.3 * nrow),
                             squeeze=False)
    flat = axes.ravel()
    for ax in flat[len(pairs):]:
        ax.axis("off")
    handles, labels = [], []
    _, color, mk = STYLE["caption"]
    for ax, pair in zip(flat, pairs):
        lo, hi, k_max = np.inf, -np.inf, 0
        for meth, rank, ls, fill in (("SMER", "smer", "-", color),
                                     ("LIME", "lime", (0, (4, 2)), "none")):
            g = df[(df.pair == pair) & (df.ranking == rank)].sort_values("k")
            if g.empty:
                continue
            xs = [0] + [int(v) for v in g["k"]]
            ys = [0.0] + [float(v) for v in g["mean"]]
            ln, = ax.plot(xs, ys, color=color, linestyle=ls,
                          marker=mk, markersize=4.0, markerfacecolor=fill,
                          markeredgecolor=color, markeredgewidth=1.0,
                          linewidth=1.1, label=meth)
            k_max = max(k_max, int(g["k"].max()))
            lo = min(lo, min(ys))
            hi = max(hi, max(ys))
            if meth not in labels:
                labels.append(meth); handles.append(ln)
        ax.set_title(pair.replace("_", " / "), fontsize=9.5 * fs)
        ax.set_xlabel(xlabel or X_LABEL, fontsize=8.5 * fs)
        ax.set_xticks(list(range(0, k_max + 1)))
        ax.set_ylim(0.0, hi * 1.08 or 0.001)
        ax.grid(alpha=0.3, linewidth=0.5)
        ax.tick_params(labelsize=7.5 * fs)
    for r in range(nrow):
        flat[r * ncols].set_ylabel(Y_LABEL, fontsize=9 * fs)
    fig.legend(handles, labels, loc="lower center",
               ncol=min(4, len(labels)), fontsize=8 * fs, frameon=False,
               bbox_to_anchor=(0.5, -0.10))
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}.pdf")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", action="append")
    ap.add_argument("--grid", metavar="SET",
                    help="one figure with every pair as a panel, for this set")
    ap.add_argument("--setup", type=int, default=5,
                    help="caption length for --grid (default 5)")
    ap.add_argument("--lime", default=LIME_RANKING,
                    choices=["lime", "lime_bow"],
                    help="which LIME ranking to plot")
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--xlabel", default=None,
                    help="override the x-axis label (tag arms remove tags, "
                         "not words)")
    ap.add_argument("--smer-only", action="store_true",
                    help="plot SMER alone; use when LIME coincides with it")
    ap.add_argument("--fontscale", type=float, default=0.85)
    ap.add_argument("--gpt", action="store_true",
                    help="AOPC for the GPT-4o configuration (aopc_gpt.csv)")
    ap.add_argument("--gpt-source", choices=["tebol", "original"],
                    default="tebol",
                    help="curves for --gpt: the stage-3 gpt4o arm (default) or "
                         "the original run's stored values in aopc_gpt.csv")
    ap.add_argument("--gpt-style", choices=["published", "qwen", "both"],
                    default="published",
                    help="layout for --gpt: the published one, the Qwen grid "
                         "styling, or both files")
    ap.add_argument("--budget", action="store_true",
                    help="AOPC deficit vs LIME's perturbation budget, "
                         "over lengths 3-10")
    ap.add_argument("--arm", default="caption",
                    help="arm for --budget (default caption)")
    ap.add_argument("--keep-old", action="store_true",
                    help="do not clear previously generated figures first")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(AOPC_CSV)
    pairs = args.pair or sorted(df["pair"].unique())

    if args.gpt:
        if args.gpt_source == "original":
            g = pd.read_csv(GPT_CSV)
        else:
            # the GPT-4o arm, trained and explained by the stage-3 pipeline
            g = df[(df.arm == "gpt4o")
                   & df.ranking.isin(["smer_subsample", "lime"])].copy()
            g["ranking"] = g["ranking"].map({"smer_subsample": "smer",
                                             "lime": "lime"})
            g["source"] = "tebol"
        ps = sorted(g["pair"].unique())
        if args.gpt_style in ("published", "both"):
            gpt_figure(g, ps, OUT / "grid_gpt", plt,
                       args.cols, args.fontscale, args.xlabel)
        if args.gpt_style in ("qwen", "both"):
            if args.gpt_source == "original":
                gpt_figure_qwen(g, ps, OUT / "grid_gpt4o", plt,
                                args.cols, args.fontscale, args.xlabel)
            else:
                # the stage-3 gpt4o arm has 25 fits, so it is drawn by the same
                # function as the Qwen3-VL grids, bands and all
                grid_figure(df, ps, ["gpt4o"], "w07", OUT / "grid_gpt4o", plt,
                            args.lime, args.cols, args.fontscale,
                            xlabel=args.xlabel, overlay=True)
        return

    if args.budget:
        budget_figure(df, pairs, args.arm,
                      OUT / f"budget_{args.arm}", plt, args.fontscale)
        return

    if args.grid:
        arms = SETS[args.grid][0]
        grid_figure(df, pairs, arms, f"w{args.setup:02d}",
                    OUT / (f"grid_{args.grid}_" +
                           ("smer" if args.smer_only else args.lime)), plt,
                    args.lime, args.cols, args.fontscale,
                    args.smer_only, args.xlabel)
        return

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

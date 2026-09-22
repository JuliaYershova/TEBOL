#!/usr/bin/env python3
"""Stage 3m -- accuracy against caption length: mean and 95% CI only.

The default is the reading version: one line per arm, its CI as a band,
nothing else. Every arm is on the same panel so the gap between them -- what
the class name is worth, what the tags are worth -- is a vertical distance you
can read off directly.

`--fits` draws the 25 individual fits faintly behind each curve, which answers
"do the fits agree, or is the shape an artefact of averaging?". The fits are
*paired* -- the same (repeat, fold) split is scored at every length -- so a fit
is a line across the whole axis rather than unrelated points. Use it with one
arm; behind four overlaid curves it is unreadable.

The `tags` arm has no length, so it enters as a horizontal reference line: the
accuracy reachable from the tag list alone, with no caption at all.

Where the curve is optimal is reported as numbers, not drawn on the chart.
`--mark` overlays the picks if you want them visible. Two picks are given per
curve: the argmax, and the shortest length within `--delta` accuracy points of
it, which is the selection rule this project uses -- it is shape-agnostic, so
flat and peaked curves need no special case.

    python scripts/21_accuracy_vs_length.py
    python scripts/21_accuracy_vs_length.py --fits
    python scripts/21_accuracy_vs_length.py --arms caption caption_noclass --tags
    python scripts/21_accuracy_vs_length.py --metric f1_macro --level caption
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

M = ROOT / "results" / "metrics" / "stage3"
FIG = ROOT / "results" / "figures" / "accuracy_vs_length"

LENGTHS = (3, 5, 7, 10, 15, 20, 25, 30)
SETUPS = [f"w{n:02d}" for n in LENGTHS]

# the four caption arms, drawn as curves
ARMS = {"caption":                ("Full captions", "#0072B2", "o"),
        "caption_tagsub":         ("Tags substituted", "#009E73", "^"),
        "caption_noclass":        ("Class name removed", "#D55E00", "s"),
        "caption_noclass_tagsub": ("Class name removed + tags", "#CC79A7", "v")}
TAGS = ("Tags only (no caption)", "#7F7F7F")


def curve(s, pair, arm, metric):
    """mean / ci_lo / ci_hi across the eight lengths, or None if absent."""
    q = s[(s.pair == pair) & (s.arm == arm)].set_index("setup")
    if not set(SETUPS) <= set(q.index):
        return None
    q = q.loc[SETUPS]
    return (q["mean"].to_numpy(), q["ci_lo"].to_numpy(), q["ci_hi"].to_numpy())


def picks(mean, delta):
    """(argmax length, shortest length within `delta` points of the max)."""
    best = int(np.argmax(mean))
    ok = np.flatnonzero(mean >= mean[best] - delta / 100.0)
    return LENGTHS[best], LENGTHS[int(ok[0])]


def panel(ax, x, series, tags, mark, delta):
    """One pair. `series` is [(label, color, marker, mean, lo, hi), ...]."""
    lo_all, hi_all = [], []

    if tags is not None:
        m, lo, hi = tags
        ax.axhspan(lo, hi, color=TAGS[1], alpha=0.12, zorder=0)
        ax.axhline(m, color=TAGS[1], linestyle="--", linewidth=1.1,
                   zorder=1, label=TAGS[0])
        lo_all.append(lo)
        hi_all.append(hi)

    for label, color, marker, mean, lo, hi, fits in series:
        if fits is not None:
            for f in fits:               # one faint line per (repeat, fold)
                ax.plot(x, f, color=color, alpha=0.16, linewidth=0.7, zorder=1)
            lo_all.append(min(f.min() for f in fits))
            hi_all.append(max(f.max() for f in fits))
        ax.fill_between(x, lo, hi, color=color, alpha=0.20, linewidth=0, zorder=2)
        ax.plot(x, mean, color=color, marker=marker, markersize=4.2,
                linewidth=1.9, markeredgecolor="white", markeredgewidth=0.6,
                zorder=3, label=label)
        lo_all.append(lo.min())
        hi_all.append(hi.max())
        if mark:
            _, short = picks(mean, delta)
            j = LENGTHS.index(short)
            ax.plot(x[j], mean[j], marker="o", markersize=9, markerfacecolor="none",
                    markeredgecolor=color, markeredgewidth=1.6, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(LENGTHS, fontsize=7.5)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    top, bot = min(1.0, max(hi_all)), min(lo_all)
    pad = (top - bot) * 0.07 or 0.003
    ax.set_ylim(bot - pad, min(1.0, top + pad))


def save(fig, stem):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote {FIG / f'{stem}.png'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", default="accuracy")
    ap.add_argument("--level", default="image")
    ap.add_argument("--delta", type=float, default=1.0,
                    help="tolerance in accuracy POINTS for the shortest-within pick")
    ap.add_argument("--mark", action="store_true",
                    help="ring the shortest-within-delta length on each curve")
    ap.add_argument("--sharey", action="store_true",
                    help="one y range for all pairs (default: per-pair range)")
    ap.add_argument("--arms", nargs="*", default=["caption"],
                    help=f"any of: {', '.join(ARMS)}")
    ap.add_argument("--fits", action="store_true",
                    help="draw the 25 individual fits faintly behind each curve")
    ap.add_argument("--tags", action="store_true",
                    help="draw the tags-only reference line (it is mostly the "
                         "class name, so it is off by default)")
    args = ap.parse_args()
    for arm in args.arms:
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm!r}; choose from {', '.join(ARMS)}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = pd.read_csv(M / "summary.csv")
    s = s[(s.level == args.level) & (s.metric == args.metric)]
    folds = None
    if args.fits:
        folds = pd.read_csv(M / "folds.csv")
        folds = folds[(folds.level == args.level) & folds.setup.isin(SETUPS)]
    if s.empty:
        raise SystemExit(f"no rows for metric={args.metric} level={args.level}")

    pairs = sorted(s[s.arm == "caption"].pair.unique())
    x = np.array(LENGTHS, dtype=float)
    xlab = "Caption length (words)"
    ylab = f"{args.metric.replace('_', ' ')} ({args.level} level)"

    stem = "_".join(args.arms) if len(args.arms) < 3 else "all_arms"
    if args.fits:
        stem += "_fits"
    cells, rows = {}, []
    for pair in pairs:
        series = []
        for arm in args.arms:
            label, color, marker = ARMS[arm]
            c = curve(s, pair, arm, args.metric)
            if c is None:
                print(f"skip {pair}/{arm}: incomplete length grid")
                continue
            mean, lo, hi = c
            fits = None
            if folds is not None:
                g = folds[(folds.pair == pair) & (folds.arm == arm)]
                fits = g.pivot_table(index=["repeat", "fold"], columns="setup",
                                     values=args.metric)[SETUPS].to_numpy()
            series.append((label, color, marker, mean, lo, hi, fits))
            best, short = picks(mean, args.delta)
            for w, m, a, b in zip(LENGTHS, mean, lo, hi):
                rows.append({"pair": pair, "arm": arm, "words": w,
                             "mean": m, "ci_lo": a, "ci_hi": b,
                             "argmax_words": best, "pick_words": short,
                             "cost_points": 100 * (mean.max() - m)})
        t = s[(s.pair == pair) & (s.arm == "tags")]
        tags = None if (t.empty or not args.tags) else (
            t.iloc[0]["mean"], t.iloc[0]["ci_lo"], t.iloc[0]["ci_hi"])
        cells[pair] = (series, tags)

    out = pd.DataFrame(rows)
    k = int(s["k"].iloc[0])
    M.mkdir(parents=True, exist_ok=True)
    out.to_csv(M / f"{args.metric}_vs_length.csv", index=False)
    print(f"wrote {M / f'{args.metric}_vs_length.csv'}")

    # ---- all pairs, all arms, one row ----
    fig, axes = plt.subplots(1, len(pairs), figsize=(3.3 * len(pairs), 3.5),
                             sharey=args.sharey)
    for ax, pair in zip(np.atleast_1d(axes), pairs):
        panel(ax, x, *cells[pair], args.mark, args.delta)
        ax.set_title(pair.replace("_", " / "), fontsize=9.5)
        ax.set_xlabel(xlab, fontsize=8.5)
    np.atleast_1d(axes)[0].set_ylabel(ylab, fontsize=9)
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    if len(l) > 1:
        fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=8.5,
                   frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(f"{args.metric.replace('_', ' ').title()} by caption length",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    save(fig, f"{args.metric}_{stem}")
    plt.close(fig)

    # ---- one file per pair ----
    for pair in pairs:
        fig, ax = plt.subplots(figsize=(4.8, 3.6))
        panel(ax, x, *cells[pair], args.mark, args.delta)
        ax.set_title(pair.replace("_", " / "), fontsize=10)
        ax.set_xlabel(xlab, fontsize=9)
        ax.set_ylabel(ylab, fontsize=9)
        if len(ax.get_legend_handles_labels()[1]) > 1:
            ax.legend(fontsize=7.5, frameon=False, loc="best")
        fig.tight_layout()
        save(fig, f"{args.metric}_{stem}_{pair}")
        plt.close(fig)

    # ---- one row per arm, so each arm can be read on its own ----
    if len(args.arms) < 2:
        return
    fig, axes = plt.subplots(len(args.arms), len(pairs), squeeze=False,
                             figsize=(2.9 * len(pairs), 2.5 * len(args.arms)),
                             sharex=True)
    for i, arm in enumerate(args.arms):
        label, color, marker = ARMS[arm]
        for j, pair in enumerate(pairs):
            ax = axes[i, j]
            series, tags = cells[pair]
            one = [t for t in series if t[0] == label]
            if not one:
                ax.axis("off")
                continue
            # tags omitted here: it sits far above the ablated arms and
            # would flatten the curve. the overlay figure carries that gap.
            panel(ax, x, one, None, args.mark, args.delta)
            if i == 0:
                ax.set_title(pair.replace("_", " / "), fontsize=9.5)
            if i == len(args.arms) - 1:
                ax.set_xlabel(xlab, fontsize=8.5)
            if j == 0:
                ax.set_ylabel(label, fontsize=8.5)
    fig.suptitle(f"{args.metric.replace('_', ' ').title()} by caption length, "
                 f"arm by arm — note the y range differs per panel",
                 fontsize=10.5, y=1.0)
    fig.tight_layout()
    save(fig, f"{args.metric}_by_arm_grid")
    plt.close(fig)

    # ---- the numbers ----
    print(f"\nshortest length within {args.delta:g} point(s) of the best "
          f"(argmax in brackets), {args.level} level:\n")
    t = out.drop_duplicates(["pair", "arm"])
    w = max(len(p) for p in pairs) + 2
    print(" " * w + "".join(f"{ARMS[a][0][:22]:>24}" for a in ARMS))
    for pair in pairs:
        line = f"{pair:<{w}}"
        for arm in ARMS:
            r = t[(t.pair == pair) & (t.arm == arm)]
            line += f"{'-':>24}" if r.empty else \
                f"{f'{r.iloc[0].pick_words:g}w  [{r.iloc[0].argmax_words:g}w]':>24}"
        print(line)


if __name__ == "__main__":
    main()

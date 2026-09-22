#!/usr/bin/env python3
"""Stage 3l -- choosing a caption length.

The obvious tool here is the elbow method, and it is the wrong one. An elbow
looks for the point of diminishing returns on a curve that keeps improving;
accuracy against caption length is not that shape. Every pair and every arm has
an interior maximum -- accuracy rises, peaks, and falls -- so there is a real
argmax to find rather than a bend to guess at.

Two rules are reported:

    best     the length with the highest mean accuracy
    1-SE     the shortest length whose mean is within one standard error of the
             best. Standard model selection, and it encodes the preference that
             actually applies here: shorter captions cost less to generate and
             less to embed, so a length that is statistically indistinguishable
             from the best but half the size is the better choice.

The standard error is over the 25 fits (5 folds x 5 repeats), the same spread
the confidence intervals elsewhere are built from.

**Two objectives disagree, and the report says so.** The length that classifies
best is not the length that separates the explainers best: accuracy peaks early
(w05 for every pair on the unablated captions) while the SMER-LIME AOPC gap
grows monotonically to w30. "Best" is only defined once you say best for what.

    python scripts/14_caption_length.py
    python scripts/14_caption_length.py --metric f1_macro
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

M = ROOT / "results" / "metrics"
FIG = ROOT / "results" / "figures" / "caption_length"

LENGTHS = (3, 5, 7, 10, 15, 20, 25, 30)
SETUPS = [f"w{n:02d}" for n in LENGTHS]
ARMS = {"caption": ("Captions", "#0072B2", "o"),
        "caption_noclass": ("Captions, synonyms removed", "#D55E00", "s")}


def select(df: pd.DataFrame) -> dict:
    """argmax and the one-standard-error choice, over lengths."""
    df = df.sort_values("o")
    m = df["mean"].to_numpy()
    se = (df["sd"] / np.sqrt(df["k"])).to_numpy()
    i = int(m.argmax())
    threshold = m[i] - se[i]
    within = np.flatnonzero(m >= threshold)
    j = int(within[0])                      # shortest that clears the bar
    return {"best": SETUPS[int(df.iloc[i]["o"])], "best_acc": m[i],
            "one_se": SETUPS[int(df.iloc[j]["o"])], "one_se_acc": m[j],
            "threshold": threshold}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", default="accuracy")
    ap.add_argument("--level", default="image")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = pd.read_csv(M / "summary.csv")
    s = s[(s.level == args.level) & (s.metric == args.metric)
          & s.setup.isin(SETUPS) & s.arm.isin(ARMS)]
    s["o"] = s.setup.map({v: i for i, v in enumerate(SETUPS)})
    pairs = sorted(s.pair.unique())

    rows = []
    for (pair, arm), g in s.groupby(["pair", "arm"]):
        r = select(g)
        r.update(pair=pair, arm=arm)
        rows.append(r)
    sel = pd.DataFrame(rows)
    sel.to_csv(M / "caption_length_selection.csv", index=False)

    # --- figure: accuracy against length, both arms, one panel per pair ---
    fig, axes = plt.subplots(1, len(pairs), figsize=(3.0 * len(pairs), 3.0),
                             sharey=False)
    for ax, pair in zip(np.atleast_1d(axes), pairs):
        for arm, (label, color, mk) in ARMS.items():
            g = s[(s.pair == pair) & (s.arm == arm)].sort_values("o")
            if g.empty:
                continue
            ax.plot(LENGTHS, g["mean"], color=color, marker=mk, markersize=4,
                    linewidth=1.3, label=label)
            ax.fill_between(LENGTHS, g["ci_lo"], g["ci_hi"], color=color,
                            alpha=0.15, linewidth=0)
            r = sel[(sel.pair == pair) & (sel.arm == arm)].iloc[0]
            ax.axvline(LENGTHS[SETUPS.index(r["one_se"])], color=color,
                       linestyle=":", linewidth=1.0, alpha=0.7)
        ax.set_title(pair.replace("_", " / "), fontsize=9)
        ax.set_xlabel("Caption length (words)", fontsize=9)
        ax.set_xticks(LENGTHS)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3, linewidth=0.5)
    np.atleast_1d(axes)[0].set_ylabel(f"{args.metric} ({args.level} level)",
                                      fontsize=9)
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{args.metric}_vs_length.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)

    print(f"{'pair':<24}{'arm':<28}{'best':>6}{'acc':>8}"
          f"{'1-SE':>7}{'acc':>8}")
    for r in sel.sort_values(["arm", "pair"]).itertuples():
        print(f"{r.pair:<24}{ARMS[r.arm][0]:<28}{r.best:>6}{r.best_acc:8.3f}"
              f"{r.one_se:>7}{r.one_se_acc:8.3f}")
    print(f"\nwrote {M / 'caption_length_selection.csv'}")
    print(f"wrote {FIG / f'{args.metric}_vs_length.png'}")


if __name__ == "__main__":
    main()

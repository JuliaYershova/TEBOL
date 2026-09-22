#!/usr/bin/env python3
"""Stage 3l -- proving why accuracy changes with caption length.

Accuracy against leakage share is confounded: leakage share is a decreasing
function of length, so that correlation is the length trend redrawn. This
removes the confound by splitting *within* each length, where every caption was
generated under the same word budget:

    captions that name their own class   vs   captions that do not

Leakage varies, length is held fixed, so the comparison is clean. The
out-of-fold predictions already exist, so nothing is retrained -- the join is
verified against summary.csv before anything is reported.

The decomposition then asks which of two things moves the overall curve:

    within   the two groups get better or worse at their own task
    mix      the share of captions naming their class changes

They point in opposite directions here, which is the whole result: the naming
*rate* rises with length, so the mix alone would push accuracy up; the overall
curve falls anyway, because captions that name their class stop being an easy
win as the class word is diluted among more tokens.

    python scripts/19_leakage_within_length.py
"""

from __future__ import annotations

import argparse
import ast
import glob
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

M = ROOT / "results" / "metrics"
FIG = ROOT / "results" / "figures" / "leakage_within_length"
MODELS = ROOT / "artifacts" / "models" / "local__qwen3-embedding__4b"

LENGTHS = (3, 5, 7, 10, 15, 20, 25, 30)
C_ALL, C_NAMED, C_UNNAMED = "#0072B2", "#009E73", "#CC79A7"


def as_list(v):
    return v if isinstance(v, list) else ast.literal_eval(v)


def caption_meta(pair: str) -> pd.DataFrame:
    """(stem, rep, n_words) -> did the original caption name its own class?"""
    files = glob.glob(str(ROOT / "artifacts" / "captions" / pair / "*--noclass.jsonl.gz"))
    rows = []
    for line in gzip.open(files[0], "rt"):
        r = json.loads(line)
        rows.append((r["stem"], int(r["rep"]), int(r["n_words"]),
                     len(as_list(r["class_tokens_removed"])) > 0))
    return pd.DataFrame(rows, columns=["stem", "rep", "n_words", "names_class"])


def ci(v):
    v = np.asarray(v, float)
    se = v.std(ddof=1) / np.sqrt(v.size)
    h = se * sps.t.ppf(0.975, df=v.size - 1)
    return v.mean(), v.mean() - h, v.mean() + h


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="verify the join against summary.csv and stop")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summ = pd.read_csv(M / "summary.csv")
    summ = summ[(summ.level == "caption") & (summ.metric == "accuracy")]
    pairs = sorted(Path(p).name for p in glob.glob(str(ROOT / "artifacts" / "captions" / "*")))

    rows = []
    for pair in pairs:
        meta = caption_meta(pair)
        for w in LENGTHS:
            f = MODELS / pair / "caption" / f"w{w:02d}" / "oof.parquet"
            if not f.exists():
                print(f"skip {pair} w{w:02d}: no oof.parquet")
                continue
            o = pd.read_parquet(f)
            pos = sorted(o.y_true.unique())[1]
            m = o.merge(meta[meta.n_words == w], on=["stem", "rep"], how="inner")
            if len(m) != len(o):
                # captions the VLM failed to produce (API timeout -> caption None)
                # are absent from the ablated artifact but still sit in this arm's
                # predictions. Too few to matter, too silent to leave unsaid.
                lost = len(o) - len(m)
                if lost > 0.001 * len(o):
                    raise SystemExit(f"{pair} w{w:02d}: joined {len(m)} of {len(o)}")
                print(f"  {pair} w{w:02d}: dropped {lost} unjoinable rows "
                      f"({lost / len(o):.4%}, failed generations)")
            m["correct"] = (m.p_pos > 0.5) == (m.y_true == pos)
            g = m.groupby(["repeat", "fold", "names_class"]).correct.mean().unstack()
            allf = m.groupby(["repeat", "fold"]).correct.mean().to_numpy()
            # the join must reproduce the number the pipeline already reported
            ref = summ[(summ.pair == pair) & (summ.arm == "caption")
                       & (summ.setup == f"w{w:02d}")]
            if not ref.empty and abs(allf.mean() - ref.iloc[0]["mean"]) > 5e-4:
                raise SystemExit(f"{pair} w{w:02d}: {allf.mean():.4f} != "
                                 f"{ref.iloc[0]['mean']:.4f} in summary.csv")
            a, alo, ahi = ci(g[True].to_numpy())
            b, blo, bhi = ci(g[False].to_numpy())
            c, clo, chi = ci(allf)
            rows.append(dict(pair=pair, n_words=w, rate=m.names_class.mean(),
                             acc_named=a, named_lo=alo, named_hi=ahi,
                             acc_unnamed=b, unnamed_lo=blo, unnamed_hi=bhi,
                             acc_all=c, all_lo=clo, all_hi=chi,
                             n_named=int(m.names_class.sum()),
                             n_unnamed=int((~m.names_class).sum())))
    d = pd.DataFrame(rows)
    print(f"join verified against summary.csv for {len(d)} cells")
    if args.check:
        return
    d.to_csv(M / "leakage_within_length.csv", index=False)
    print(f"wrote {M / 'leakage_within_length.csv'}")
    FIG.mkdir(parents=True, exist_ok=True)

    # --- the split, one panel per pair ---
    fig, axes = plt.subplots(1, len(pairs), figsize=(3.1 * len(pairs), 3.5))
    for ax, pair in zip(np.atleast_1d(axes), pairs):
        g = d[d.pair == pair].sort_values("n_words")
        x = g.n_words.to_numpy()
        ax.axhline(0.5, color="#898781", linewidth=1, zorder=1)
        ax.annotate("chance", (30, 0.5), fontsize=7, color="#898781",
                    xytext=(0, 3), textcoords="offset points", ha="right")
        for col, lo, hi, color, mk, lab in [
                ("acc_named", "named_lo", "named_hi", C_NAMED, "^", "names its class"),
                ("acc_all", "all_lo", "all_hi", C_ALL, "o", "all captions"),
                ("acc_unnamed", "unnamed_lo", "unnamed_hi", C_UNNAMED, "v", "does not name it")]:
            ax.plot(x, g[col], color=color, marker=mk, markersize=4, linewidth=1.7,
                    label=lab, zorder=3)
            ax.fill_between(x, g[lo], g[hi], color=color, alpha=0.18, linewidth=0, zorder=2)
        ax.set_title(pair.replace("_", " / "), fontsize=9)
        ax.set_xlabel("Caption length (words)", fontsize=8.5)
        ax.set_xticks(LENGTHS)
        ax.set_ylim(0.45, 1.02)
        ax.tick_params(labelsize=7.5)
        ax.grid(alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)
    np.atleast_1d(axes)[0].set_ylabel("accuracy (caption level)", fontsize=9)
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Same length, leakage varying — captions that name their class "
                 "vs captions that do not", fontsize=10, y=1.02)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"split_by_leakage.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG / 'split_by_leakage.png'}")

    # --- decomposition: what actually moves the overall curve, 3 -> 30 words ---
    print(f"\n{'pair':<24}{'total 3->30':>13}{'within-group':>14}{'mix':>9}{'interaction':>13}")
    dec = []
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    for i, pair in enumerate(pairs):
        g = d[d.pair == pair].sort_values("n_words")
        a0, a1 = g.acc_named.iloc[0], g.acc_named.iloc[-1]
        b0, b1 = g.acc_unnamed.iloc[0], g.acc_unnamed.iloc[-1]
        w0, w1 = g.rate.iloc[0], g.rate.iloc[-1]
        base = w0 * a0 + (1 - w0) * b0
        total = (w1 * a1 + (1 - w1) * b1) - base
        within = (w0 * a1 + (1 - w0) * b1) - base
        mix = (w1 * a0 + (1 - w1) * b0) - base
        dec.append(dict(pair=pair, total=total, within=within, mix=mix,
                        interaction=total - within - mix))
        print(f"{pair:<24}{total*100:>12.1f}{within*100:>14.1f}"
              f"{mix*100:>9.1f}{(total-within-mix)*100:>13.1f}")
        ax.bar(i - 0.21, within * 100, 0.4, color=C_NAMED, zorder=3,
               label="within-group" if i == 0 else None)
        ax.bar(i + 0.21, mix * 100, 0.4, color=C_UNNAMED, zorder=3,
               label="mix (naming rate)" if i == 0 else None)
        ax.plot(i, total * 100, marker="D", markersize=6, color=C_ALL, zorder=4,
                linestyle="none", label="total change" if i == 0 else None)
    pd.DataFrame(dec).to_csv(M / "leakage_decomposition.csv", index=False)
    ax.axhline(0, color="#c3c2b7", linewidth=1)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels([p.replace("_", " /\n") for p in pairs], fontsize=8)
    ax.set_ylabel("change in accuracy, 3 → 30 words\n(percentage points)", fontsize=9)
    ax.set_title("What moves the curve: the groups themselves, not the mix", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"decomposition.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {FIG / 'decomposition.png'}")
    print(f"wrote {M / 'leakage_decomposition.csv'}")


if __name__ == "__main__":
    main()

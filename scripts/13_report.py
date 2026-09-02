#!/usr/bin/env python3
"""Stage 3k -- assemble the stage-3 results into one readable document.

Reads what the earlier stages wrote and produces results/stage3_report.md:
classifier accuracy, the words each explainer says the classifier uses, how
faithful those words are under AOPC, and how much of any of it survives a
rerun. Nothing is recomputed here; if a number looks wrong, it is wrong in the
stage that produced it.

Written as a script rather than a hand-edited file so it tracks the data. Every
table carries the interval it was measured with, because most of the
differences in this study are smaller than they look.

    python scripts/13_report.py
    python scripts/13_report.py --setup 10 --topn 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tebol import arms as arms_mod  # noqa: E402

M = ROOT / "results" / "metrics" / "stage3"
OUT = ROOT / "results" / "stage3_report.md"

PAIRS = ("acousticguitar_violin", "ambulance_firetruck", "ant_bee",
         "cucumber_zucchini", "hotpot_vase")
ARM_LABEL = {
    "caption": "Captions",
    "caption_noclass": "Captions, synonyms removed",
    "tags": "ImageNet tags",
    "caption_tagsub": "Captions (tag subset)",
    "caption_noclass_tagsub": "Captions, synonyms removed (tag subset)",
}


def fmt(m, lo, hi, dp=3):
    return f"{m:.{dp}f} <sub>{lo:.{dp}f}–{hi:.{dp}f}</sub>"


def accuracy_section(setup: str) -> list[str]:
    s = pd.read_csv(M / "summary.csv")
    s = s[(s.level == "image") & (s.metric == "accuracy")]
    L = ["## 1. Classifier accuracy", "",
         "Image level: the five captions of a photograph are averaged to one "
         "probability before scoring. Mean over 25 fits (5-fold × 5 repeats) "
         "with the across-fold 95% interval.", "",
         f"### By representation (at {setup})", "",
         "| pair | " + " | ".join(ARM_LABEL[a] for a in ARM_LABEL) + " |",
         "|---|" + "---|" * len(ARM_LABEL)]
    for pair in PAIRS:
        cells = []
        for arm in ARM_LABEL:
            want = "tags" if arm == "tags" else setup
            r = s[(s.pair == pair) & (s.arm == arm) & (s.setup == want)]
            cells.append(fmt(*r.iloc[0][["mean", "ci_lo", "ci_hi"]])
                         if not r.empty else "--")
        L.append(f"| {pair} | " + " | ".join(cells) + " |")

    L += ["", "### By caption length", "",
          "Does asking the captioner for more words help?", "",
          "| pair | arm | 3 words | 5 words | 7 words | 10 words |",
          "|---|---|---|---|---|---|"]
    for pair in PAIRS:
        for arm in ("caption", "caption_noclass"):
            cells = []
            for w in ("w03", "w05", "w07", "w10"):
                r = s[(s.pair == pair) & (s.arm == arm) & (s.setup == w)]
                cells.append(f"{r['mean'].iloc[0]:.3f}" if not r.empty else "--")
            L.append(f"| {pair} | {ARM_LABEL[arm]} | " + " | ".join(cells) + " |")
    return L + [""]


def _global_words(path: Path, col: str, pos_label: str, topn: int):
    """Corpus word ranking from a per-word table, split by predicted class."""
    df = pd.read_parquet(path)
    df = df.assign(dir=np.where(df["pred_class"] == pos_label, "pos", "neg"))
    agg = df.groupby(["dir", "word"])[col].agg(["mean", "size"]).reset_index()
    agg = agg[agg["size"] >= 20]
    out = {}
    for d, asc in (("pos", False), ("neg", True)):
        sub = agg[agg["dir"] == d].sort_values("mean", ascending=asc)
        out[d] = list(sub["word"].head(topn))
    return out


def words_section(setup: str, topn: int) -> list[str]:
    L = ["## 2. Which words the classifier uses", "",
         "Corpus-level ranking: each word's score averaged over every caption "
         "it appears in, restricted to words seen in at least 20 captions. "
         "SMER ranks by `z = β·e`; LIME by its own weight, both signed toward "
         "the predicted class.", ""]
    for pair in PAIRS:
        neg, pos = pair.split("_")
        L += [f"### {pair}", ""]
        for arm in ("caption", "caption_noclass"):
            emb = arms_mod.ARMS[arm].embedding
            base = ROOT / "artifacts" / "explanations" / emb / pair / arm / setup
            sp, lp = base / "smer_words.parquet", base / "lime_words__bow.parquet"
            if not sp.exists() or not lp.exists():
                continue
            pos_label = sorted((neg, pos))[1]
            sw = _global_words(sp, "z", pos_label, topn)
            lw = _global_words(lp, "lime", pos_label, topn)
            L += [f"**{ARM_LABEL[arm]}**", "",
                  "| class | SMER | LIME | shared |", "|---|---|---|---:|"]
            for d, cls in (("pos", sorted((neg, pos))[1]),
                           ("neg", sorted((neg, pos))[0])):
                shared = len(set(sw[d]) & set(lw[d]))
                L.append(f"| {cls} | {', '.join(sw[d])} | {', '.join(lw[d])} "
                         f"| {shared}/{topn} |")
            L.append("")
    return L


def aopc_section(setup: str) -> list[str]:
    a = pd.read_csv(M / "aopc.csv")
    a = a[a.ranking.isin(["smer_subsample", "lime_bow"])]
    mx = a.groupby(["pair", "arm", "setup", "ranking"])["k"].transform("max")
    end = a[a.k == mx]
    L = ["## 3. Explanation faithfulness (AOPC)", "",
         "Mean probability decrease after removing the k highest-ranked words, "
         "at the largest k each setup allows (caption length − 1). LIME uses "
         "`LimeTextExplainer` defaults; both explainers score the same "
         "captions.", "",
         "| pair | representation | SMER | LIME | SMER − LIME |",
         "|---|---|---|---|---:|"]
    for pair in PAIRS:
        for arm in ARM_LABEL:
            want = "tags" if arm == "tags" else setup
            s = end[(end.pair == pair) & (end.arm == arm) & (end.setup == want)
                    & (end.ranking == "smer_subsample")]
            l = end[(end.pair == pair) & (end.arm == arm) & (end.setup == want)
                    & (end.ranking == "lime_bow")]
            if s.empty or l.empty:
                continue
            sm, lm = s["mean"].iloc[0], l["mean"].iloc[0]
            L.append(f"| {pair} | {ARM_LABEL[arm]} "
                     f"| {fmt(sm, s['ci_lo'].iloc[0], s['ci_hi'].iloc[0])} "
                     f"| {fmt(lm, l['ci_lo'].iloc[0], l['ci_hi'].iloc[0])} "
                     f"| {sm - lm:+.4f} |")
    return L + [""]


def stability_section() -> list[str]:
    L = ["## 4. How much survives a rerun", "",
         "Mean pairwise Jaccard overlap of the top-3 words, and how often the "
         "single most important word changes. Three sources of randomness, "
         "varied one at a time.", "",
         "| pair | arm | source | method | top-3 Jaccard | top-1 changes |",
         "|---|---|---|---|---|---|"]
    sm = pd.read_csv(M / "stochasticity.csv")
    lm = pd.read_csv(M / "stochasticity_lime.csv")
    sm["method"] = "SMER"
    both = pd.concat([sm, lm], ignore_index=True)
    order = {"model": 0, "caption": 1, "seed": 2}
    both["o"] = both["source"].map(order)
    both = both.sort_values(["pair", "arm", "o", "method"])
    for r in both.itertuples():
        t1 = "--" if pd.isna(r.top1_changed) else f"{r.top1_changed:.3f}"
        L.append(f"| {r.pair} | {ARM_LABEL.get(r.arm, r.arm)} | {r.source} "
                 f"| {r.method} | {r.jaccard:.3f} | {t1} |")
    L += ["", "`model` = refit on a different cross-validation split; "
          "`caption` = the captioner asked again at temperature 1.0; "
          "`seed` = LIME's `random_state`. SMER has no `seed` row of its own "
          "beyond the derived 1.000: `z = β·e` involves no sampling.", ""]
    p = pd.read_csv(M / "prediction_stability.csv")
    L += ["**Prediction stability** — share of images whose five captions all "
          "agree on the predicted class:", "",
          "| pair | " + " | ".join(sorted(p.arm.unique())) + " |",
          "|---|" + "---|" * p.arm.nunique()]
    for pair in PAIRS:
        cells = [f"{p[(p.pair == pair) & (p.arm == a)]['unanimous_images'].iloc[0]:.3f}"
                 if not p[(p.pair == pair) & (p.arm == a)].empty else "--"
                 for a in sorted(p.arm.unique())]
        L.append(f"| {pair} | " + " | ".join(cells) + " |")
    return L + [""]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setup", default="w07")
    ap.add_argument("--topn", type=int, default=10)
    args = ap.parse_args()

    lines = [
        "# Stage 3 — classifier, explanations, and how much of it is stable",
        "",
        f"All tables at **{args.setup}** unless stated. Five class pairs, "
        "logistic regression on mean-pooled qwen3-embedding vectors (2560-d), "
        "5-fold × 5-repeat cross-validation with folds assigned to *images* so "
        "the five captions of one photograph never straddle a split.",
        "",
        "Generated by `scripts/13_report.py` from the stage-3 metrics; rerun it "
        "after any stage changes.",
        "",
        *accuracy_section(args.setup),
        *words_section(args.setup, args.topn),
        *aopc_section(args.setup),
        *stability_section(),
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT} ({len(lines)} lines)")


if __name__ == "__main__":
    main()

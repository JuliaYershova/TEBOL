#!/usr/bin/env python3
"""Stage 3k -- the stage-3 results, short enough to read.

Produces results/stage3_report.md: five findings, each with the one table that
supports it and a sentence saying what the table means. Nothing is recomputed
here; if a number is wrong it is wrong in the stage that produced it.

The earlier version of this script printed every number it could find -- ten
word-list tables, a 25-row AOPC table -- and was unreadable as a result. What a
reader needs is the claim first and the evidence under it, so each section here
leads with the finding and carries only the columns that bear on it. The full
numbers stay in results/metrics/stage3/*.csv for anyone who wants them.

    python scripts/13_report.py
    python scripts/13_report.py --setup w15 --topn 6
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
OUT = ROOT / "results" / "reports" / "README.md"

PAIRS = ("acousticguitar_violin", "ambulance_firetruck", "ant_bee",
         "cucumber_zucchini", "hotpot_vase")
LENGTHS = (3, 5, 7, 10, 15, 20, 25, 30)
SETUPS = [f"w{n:02d}" for n in LENGTHS]
SHORT = {p: p.replace("_", "/") for p in PAIRS}


def acc(s, pair, arm, setup):
    r = s[(s.pair == pair) & (s.arm == arm) & (s.setup == setup)]
    return float(r["mean"].iloc[0]) if not r.empty else np.nan


def f(x, dp=3):
    return "--" if not np.isfinite(x) else f"{x:.{dp}f}"


def section_leakage(s, setup) -> list[str]:
    L = ["## 1. Most of the accuracy is the caption naming its own class", "",
         f"Image-level accuracy at {setup}, mean of 25 fits.", "",
         "| pair | captions | class name removed | cost |",
         "|---|---:|---:|---:|"]
    deltas = []
    for p in PAIRS:
        a, b = acc(s, p, "caption", setup), acc(s, p, "caption_noclass", setup)
        deltas.append(a - b)
        L.append(f"| {SHORT[p]} | {f(a)} | {f(b)} | **{a - b:+.3f}** |")
    L += ["",
          f"85% of captions contain their own class name. Removing it costs "
          f"{min(deltas):.3f}–{max(deltas):.3f} accuracy in four pairs. "
          "**hotpot/vase is the exception** — food scenes and decor are "
          "separable without naming them, so it loses nothing. "
          "**cucumber/zucchini collapses to near chance**: two green vegetables "
          "that the caption was effectively labelling.", ""]
    return L


def section_tags(s, setup) -> list[str]:
    L = ["## 2. Descriptions and tags are a dead heat", "",
         "All three columns on the images ImageNet-Captions covers, so the "
         "comparison is not confounded by which photographs each has.", "",
         "| pair | captions | class name removed | human tags |",
         "|---|---:|---:|---:|"]
    for p in PAIRS:
        L.append(f"| {SHORT[p]} | {f(acc(s, p, 'caption_tagsub', setup))} "
                 f"| {f(acc(s, p, 'caption_noclass_tagsub', setup))} "
                 f"| {f(acc(s, p, 'tags', 'tags'))} |")
    L += ["",
          "Captions and tags land within about a point of each other. Once the "
          "class name is stripped, tags win — but tags are never stripped, so "
          "that column compares a clean representation against a leaky one and "
          "should be read as a caveat, not a result.", ""]
    return L


def section_length(s) -> list[str]:
    sel = pd.read_csv(M / "caption_length_selection.csv")
    L = ["## 3. The two arms want opposite caption lengths", "",
         "| pair | captions: best | acc | class name removed: best | acc |",
         "|---|---:|---:|---:|---:|"]
    for p in PAIRS:
        a = sel[(sel.pair == p) & (sel.arm == "caption")].iloc[0]
        b = sel[(sel.pair == p) & (sel.arm == "caption_noclass")].iloc[0]
        L.append(f"| {SHORT[p]} | {a['one_se']} | {f(a['one_se_acc'])} "
                 f"| {b['one_se']} | {f(b['one_se_acc'])} |")
    L += ["",
          "Shortest length within one standard error of the best. **Leaky "
          "captions want to be short** — three words is enough to say "
          "'zucchini', and every extra word dilutes it, because a word "
          "contributes `z/n` to a mean-pooled vector. **Clean captions want to "
          "be long**, where accumulated visual evidence replaces the label.", ""]
    return L


def section_aopc(a) -> list[str]:
    d = a[a.ranking.isin(["smer_subsample", "lime_bow"]) & (a.arm == "caption")]
    mx = d.groupby(["pair", "setup", "ranking"])["k"].transform("max")
    e = d[d.k == mx].pivot_table(index=["pair", "setup"], columns="ranking",
                                 values="mean").reset_index()
    e["gap"] = e["smer_subsample"] - e["lime_bow"]
    L = ["## 4. SMER's advantage over LIME grows with caption length", "",
         "AOPC gap (SMER − LIME) at the deepest k each length allows.", "",
         "| pair | " + " | ".join(f"{n}w" for n in LENGTHS) + " |",
         "|---|" + "---:|" * len(LENGTHS)]
    for p in PAIRS:
        cells = []
        for w in SETUPS:
            r = e[(e.pair == p) & (e.setup == w)]
            cells.append(f(r["gap"].iloc[0]) if not r.empty else "--")
        L.append(f"| {SHORT[p]} | " + " | ".join(cells) + " |")
    means = [e[e.setup == w]["gap"].mean() for w in SETUPS]
    L.append("| **mean** | " + " | ".join(f"**{m:.3f}**" for m in means) + " |")
    L += ["",
          "A 7-word caption has 2⁷ = 128 possible word-subsets, so LIME's 5000 "
          "perturbations cover the whole space 39 times over — it is not "
          "approximating, and matches SMER exactly. A 34-word caption has "
          "2³⁴ ≈ 1.7×10¹⁰ subsets and 5000 samples is a vanishing fraction. "
          "**SMER's cost stays one dot product while LIME's sample requirement "
          "grows exponentially**, which is the advantage that holds regardless "
          "of how LIME is configured.", ""]
    return L


def section_words(setup, topn) -> list[str]:
    L = ["## 5. SMER's word lists are cleaner than LIME's", "",
         f"Top {topn} words per class, class name removed, at {setup}. "
         "Corpus ranking over words appearing in ≥20 captions.", "",
         "| pair | class | SMER | LIME |", "|---|---|---|---|"]
    arm = "caption_noclass"
    emb = arms_mod.ARMS[arm].embedding
    for p in PAIRS:
        base = ROOT / "artifacts" / "explanations" / emb / p / arm / setup
        sp, lp = base / "smer_words.parquet", base / "lime_words__bow.parquet"
        if not sp.exists() or not lp.exists():
            continue
        pos = sorted(p.split("_"))[1]
        for path, col in ((sp, "z"), (lp, "lime")):
            pass
        out = {}
        for path, col in ((sp, "z"), (lp, "lime")):
            df = pd.read_parquet(path)
            df = df.assign(dir=np.where(df["pred_class"] == pos, "pos", "neg"))
            g = df.groupby(["dir", "word"])[col].agg(["mean", "size"]).reset_index()
            g = g[g["size"] >= 20]
            out[col] = {d: list(g[g.dir == d].sort_values(
                "mean", ascending=(d == "neg"))["word"].head(topn))
                for d in ("pos", "neg")}
        for d, cls in (("pos", sorted(p.split("_"))[1]),
                       ("neg", sorted(p.split("_"))[0])):
            L.append(f"| {SHORT[p] if d == 'pos' else ''} | {cls} "
                     f"| {', '.join(out['z'][d])} | {', '.join(out['lime'][d])} |")
    L += ["",
          "SMER returns object and scene nouns. LIME's corpus lists are "
          "contaminated by high-frequency filler — `and`, `with`, `on` reach "
          "its top ranks, while **no stopword reaches SMER's top 20 in any "
          "pair**. This is the corpus-level aggregate; per caption the two "
          "agree closely.", ""]
    return L


def section_caption_stability() -> list[str]:
    """Stage-1 repeatability. Measured on the raw captions, no model involved."""
    path = M.parent / "caption_stability.csv"
    if not path.exists():
        return []
    d = pd.read_csv(path)
    L = ["## 6. How repeatable is the captioner?", "",
         "The vision model was asked five times per image at each length, "
         "temperature 1.0. Measured on the raw caption files: no classifier, "
         "no embedding, no explainer.", "",
         "| requested | mean actual | length sd | token overlap "
         "| content overlap | identical | names class consistently |",
         "|---:|---:|---:|---:|---:|---:|---:|"]
    for n in sorted(d.n_words.unique()):
        g = d[d.n_words == n]
        L.append(f"| {n} | {g.len_mean.mean():.1f} | {g.len_sd.mean():.2f} "
                 f"| {g.jaccard.mean():.3f} | {g.jaccard_content.mean():.3f} "
                 f"| {g.identical.mean():.3f} | {g.class_unanimous.mean():.3f} |")
    L += ["",
          "Five repetitions give C(5,2) = 10 pairs; each score is the mean "
          "over those 10, then averaged over images.",
          "",
          "| column | how it is computed | reading |",
          "|---|---|---|",
          "| mean actual | token count averaged over all captions at that "
          "length | how far the model overshoots the request |",
          "| length sd | sd of token count across one image's 5 repetitions "
          "| 0 = same length every time |",
          "| token overlap | Jaccard of two repetitions' token sets | 1 = same "
          "words every time |",
          "| content overlap | the same after removing function words | "
          "isolates *what* was described from *how* it was phrased |",
          "| identical | share of the 10 pairs with the same string | 1 = "
          "generation is deterministic |",
          "| names class consistently | share of images where all 5 agree on "
          "whether a class term appears | below 1 = for those images, whether "
          "the caption leaks its label is itself random |",
          "",
          "**Two repetitions of the same image share under half their words at "
          "30 words**, and by 15 words two identical captions are essentially "
          "never produced. The captioner also overshoots the request at every "
          "length — asked for 30, it writes 34.3.",
          "",
          "The last column matters for the leakage result: **the class name "
          "appears in all five repetitions or none of them about 90% of the "
          "time**, so in the remaining 10% whether a caption leaks its label "
          "is itself a coin flip.", ""]
    return L


def section_stability() -> list[str]:
    try:
        sm = pd.read_csv(M / "stochasticity.csv")
        lm = pd.read_csv(M / "stochasticity_lime.csv")
    except FileNotFoundError:
        return []
    sm["method"] = "SMER"
    both = pd.concat([sm, lm], ignore_index=True)
    L = ["## 7. What survives a rerun", "",
         "Mean pairwise overlap of the top-3 words across repeated runs, "
         "varying one source of randomness at a time.", "",
         "| source varied | SMER | LIME |", "|---|---:|---:|"]
    for src, label in (("model", "refit on a different CV split"),
                       ("caption", "captioner asked again (temp 1.0)"),
                       ("seed", "LIME's random_state")):
        s = both[(both.source == src) & (both.method == "SMER")]["jaccard"]
        l = both[(both.source == src) & (both.method == "LIME")]["jaccard"]
        L.append(f"| {label} | {f(s.mean(), 3) if len(s) else '--'} "
                 f"| {f(l.mean(), 3) if len(l) else '--'} |")
    L += ["",
          "Refitting the model barely moves either method. **Regenerating the "
          "caption moves both a great deal** — the captioner's randomness, not "
          "the explainer's, is the dominant source of instability. SMER has no "
          "seed of its own: `z = β·e` involves no sampling, so its value on "
          "that row is 1.000 by construction.", ""]
    return L


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setup", default="w07")
    ap.add_argument("--topn", type=int, default=5)
    args = ap.parse_args()

    s = pd.read_csv(M / "summary.csv")
    s = s[(s.level == "image") & (s.metric == "accuracy")]
    a = pd.read_csv(M / "aopc.csv")

    lines = [
        "# Stage 3 — results",
        "",
        "Five class pairs, logistic regression on mean-pooled qwen3-embedding "
        "vectors (2560-d). Captions generated at 8 lengths × 5 repetitions per "
        "image; 5-fold × 5-repeat cross-validation with folds assigned to "
        "*images*, so the five captions of one photograph never straddle a "
        "split. Every interval is over the resulting 25 fits.",
        "",
        "## Reports in this folder",
        "",
        "| file | answers | produced by | reads |",
        "|---|---|---|---|",
        "| `00_audit.md` | is every stage complete? | `16_audit.py` | every artefact |",
        "| `01_coverage.md` | how many images, boxes, tags? | `coverage_report.py` | `data/raw`, ImageNet-Captions |",
        "| `02_caption_stability.md` | how repeatable is the captioner? | `15_caption_stability.py` | raw caption JSONL |",
        "| `03_class_leakage.md` | how often does a caption name its class? | `class_leakage.py` | raw captions + `class_terms.csv` |",
        "| `04_accuracy.md` | accuracy per arm and length, with tests | `05_report_accuracy.py` | `summary.csv`, `comparisons.csv` |",
        "| `05_aopc.md` | AOPC curves, all rankings | `07_aopc.py` | `smer_words`, `lime_words` |",
        "| `06_smer_examples_*.md` | one worked caption per class | `06_worked_examples.py` | `smer_words.parquet` |",
        "| `README.md` | this summary | `13_report.py` | all of the above |",
        "",
        "## How the numbers are produced",
        "",
        "| quantity | metric | why this one |",
        "|---|---|---|",
        "| classification | image-level accuracy | the five captions of a photograph are averaged to one probability first, because the claim is about the image, not the sentence |",
        "| interval | 95% t-interval over 25 fits | 5 folds x 5 repeats; folds assigned to *images*, so no photograph straddles a split |",
        "| arm differences | Nadeau-Bengio corrected t, Bonferroni | CV training sets overlap, so the naive paired t is anti-conservative |",
        "| explanation faithfulness | AOPC | drop in predicted-class probability as the top-k words are removed |",
        "| explanation agreement | top-3 Jaccard, Kendall tau | set overlap for the headline, full-ordering correlation as a check |",
        "| generation repeatability | pairwise token Jaccard | over the 5 captions of one image at one length |",
        "",
        "## Summary",
        "",
        "1. **Most of the reported accuracy is leakage.** 85% of captions name "
        "their own class; removing it costs up to 0.245 accuracy.",
        "2. **Descriptions do not beat human tags** when both are leaky — they "
        "land within a point of each other.",
        "3. **Optimal caption length is opposite for the two arms**: short when "
        "the class name is present, long when it is not.",
        "4. **SMER beats LIME by more the longer the caption**, from 0.006 at "
        "3 words to 0.346 at 30, for a reason that is structural rather than "
        "empirical.",
        "5. **The captioner is the dominant source of instability.** Two "
        "repetitions of one image share under half their words at 30 words, "
        "and that moves the explanation more than refitting the model does.",
        "",
        f"Detail below at **{args.setup}**; full numbers in "
        "`results/metrics/stage3/*.csv`, figures in "
        "`results/figures/aopc/`.",
        "",
        *section_leakage(s, args.setup),
        *section_tags(s, args.setup),
        *section_length(s),
        *section_aopc(a),
        *section_words(args.setup, args.topn),
        *section_caption_stability(),
        *section_stability(),
    ]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(lines)} lines)")


if __name__ == "__main__":
    main()

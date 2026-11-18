#!/usr/bin/env python3
"""Stage 7 -- how exactly does SMER rebuild the model's own logit?

The paper says the per-word contributions "reconstruct the model's own logit".
That is an identity in real arithmetic,

    b + (1/n) SUM_i (w . e_i)  =  b + w . (1/n) SUM_i e_i  =  decision_function(x),

so what can be measured is the floating-point gap between the two sides. This
script measures it, once, for every configuration the paper quotes it for, so the
number in the text is a file in the repository rather than a console printout:

    qwen3vl_binary   the five binary tasks, Qwen3-VL captions of 5 words,
                     Qwen3-Embedding vectors
    gpt4o_binary     the five binary tasks, GPT-4o descriptions of 7 words,
                     ada-002 vectors (scripts/reproduce_gpt4o.py)
    multiclass_food  cucumber / zucchini / hotpot, one-vs-rest, 5-word captions
                     (scripts/25_multiclass.py), all three class logits

For each, every held-out caption of every one of the 25 fits (5 x 5 folds) is
checked -- not a sample. The left side is `bias + mean(z)` with z = vocab @ beta
as `tebol.smer` computes it (float32 vectors, float32 beta, mean taken in
float64); the right side is the fitted model's own `decision_function`.

`dtype` says what the vectors, the coefficients and the pooling were held in:

    float32   as the pipeline stores and uses them: float32 vectors, float32
              beta, float32 mean-pooled design matrix (the number to quote)
    float64   the same fitted model with vectors, coefficients and pooling all
              promoted to float64 -- if the gap collapses to ~1e-13 it was
              rounding, not approximation

    python scripts/26_decomposition_gap.py
    python scripts/26_decomposition_gap.py --only gpt4o_binary --out /tmp/gap.csv

Writes results/metrics/decomposition_gap.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tebol import arms as arms_mod  # noqa: E402
from tebol import folds as folds_mod, smer  # noqa: E402

PAIRS = ("acousticguitar_violin", "ambulance_firetruck", "ant_bee",
         "cucumber_zucchini", "hotpot_vase")
FOOD = ["cucumber", "zucchini", "hotpot"]
OUT = ROOT / "results" / "metrics" / "decomposition_gap.csv"
LR_KW = dict(max_iter=2000, class_weight="balanced")
DTYPES = ("float32", "float64")


def gaps_binary(data, x, test, clf, dtype):
    """|bias + mean(z) - decision_function| per held-out caption."""
    beta = clf.coef_.ravel()
    b = float(clf.intercept_[0])
    if dtype == "float32":
        z = smer.vocab_logits(data.store, beta.astype(np.float32))
        model = clf.decision_function(x[test])
    else:
        v = data.store.vectors.astype(np.float64)
        z = v @ beta.astype(np.float64)
        model = np.array([v[data.word_rows(i)].mean(axis=0) @ beta.astype(np.float64) + b
                          for i in test])
    return np.array([abs(b + z[data.word_rows(i)].astype(np.float64).mean() - m)
                     for i, m in zip(test, model)])


def run_binary(arm, setup):
    rows = []
    for pair in PAIRS:
        data = arms_mod.load_arm(ROOT, pair, arm, setup)
        keep = np.flatnonzero(arms_mod.nonempty(data))
        x = arms_mod.caption_means(data)
        folds = folds_mod.load_folds(ROOT / "results" / "folds" / f"{pair}.csv")
        worst = {d: 0.0 for d in DTYPES}
        n = 0
        for repeat in range(int(folds["repeat"].max()) + 1):
            assign = folds_mod.fold_of(folds, repeat)
            fold = np.array([assign.get(s, -1) for s in data.stem])
            for f in sorted(set(fold[keep])):
                test = keep[fold[keep] == f]
                train = keep[fold[keep] != f]
                clf = LogisticRegression(**LR_KW).fit(x[train], data.cls[train])
                n += len(test)
                for d in DTYPES:
                    worst[d] = max(worst[d], float(gaps_binary(data, x, test, clf, d).max()))
        for d in DTYPES:
            rows.append((pair, "all_25_fits", n, d, worst[d]))
        print(pair, arm, setup, n, worst, flush=True)
    return rows


def run_multiclass(setup):
    from sklearn.multiclass import OneVsRestClassifier
    data = arms_mod.load_classes(ROOT, FOOD, "caption", setup)
    classes = sorted(set(data.cls))
    x = arms_mod.caption_means(data)
    folds = folds_mod.load_folds(ROOT / "results" / "folds" / "multiclass_food.csv")
    worst = {d: 0.0 for d in DTYPES}
    n = 0
    for repeat in sorted(folds["repeat"].unique()):
        assign = folds_mod.fold_of(folds, int(repeat))
        fold = np.array([assign[s] for s in data.stem])
        for f in sorted(set(fold.tolist())):
            te, tr = np.flatnonzero(fold == f), np.flatnonzero(fold != f)
            clf = OneVsRestClassifier(LogisticRegression(**LR_KW)).fit(x[tr], data.cls[tr])
            beta = np.vstack([e.coef_.ravel() for e in clf.estimators_])
            bias = np.array([float(e.intercept_[0]) for e in clf.estimators_])
            n += len(te)
            for d in DTYPES:
                if d == "float32":
                    zv = data.store.vectors @ beta.astype(np.float32).T
                    model = clf.decision_function(x[te])
                else:
                    v = data.store.vectors.astype(np.float64)
                    zv = v @ beta.T
                    model = np.array([v[data.word_rows(i)].mean(axis=0) @ beta.T + bias
                                      for i in te])
                for i, m in zip(te, model):
                    got = bias + zv[data.word_rows(i)].astype(np.float64).mean(axis=0)
                    worst[d] = max(worst[d], float(np.abs(got - m).max()))
        print("multiclass repeat", repeat, worst, flush=True)
    return [("+".join(FOOD), "all_25_fits", n, d, worst[d]) for d in DTYPES]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["qwen3vl_binary", "gpt4o_binary", "multiclass_food"],
                    nargs="*", help="run a subset (default: all three)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    todo = args.only or ["qwen3vl_binary", "gpt4o_binary", "multiclass_food"]

    rows = []
    if "qwen3vl_binary" in todo:
        rows += [("qwen3vl_binary", *r[:1], f"caption_w05_{r[1]}", *r[2:]) for r in run_binary("caption", 5)]
    if "gpt4o_binary" in todo:
        rows += [("gpt4o_binary", *r[:1], f"gpt4o_w07_{r[1]}", *r[2:]) for r in run_binary("gpt4o", 7)]
    if "multiclass_food" in todo:
        rows += [("multiclass_food", r[0], f"caption_w05_{r[1]}", *r[2:]) for r in run_multiclass(5)]
    out = pd.DataFrame(rows, columns=["experiment", "pair", "setup", "n_captions",
                                      "dtype", "max_abs_gap"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.groupby(["experiment", "dtype"]).max_abs_gap.max().to_string())
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

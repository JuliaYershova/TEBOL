#!/usr/bin/env python3
"""Rebuild the GPT-4o rows of the accuracy table from the stored word vectors alone.

No API call and no captioning model is involved. The seven-word GPT-4o descriptions
were embedded word by word with text-embedding-ada-002 once, and what the classifier
needs from that run is committed in artifacts/embeddings/openai__text-embedding-ada-002/:

    vocab.npz     one 1536-d vector per distinct word
    index/        for every description, the rows of vocab.npz that make it up
    vocab.csv     the words

This script fits the paper's classifier on those vectors (logistic regression on the
mean of a description's word vectors, the same 5 x 5 cross-validation folds as every
other arm) and checks two things for every task:

    accuracy   image-level accuracy, averaged over the 25 fits, against the value in
               results/metrics/summary.csv, which is the value in the paper's table
    SMER       for every held-out description, bias + mean of the per-word logits z
               against the classifier's own decision function; the largest gap

    python scripts/reproduce_gpt4o.py

Writes results/metrics/gpt4o_reproduction.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tebol import arms as arms_mod  # noqa: E402
from tebol import evaluate, folds as folds_mod, smer  # noqa: E402

PAIRS = ("acousticguitar_violin", "ambulance_firetruck", "ant_bee",
         "cucumber_zucchini", "hotpot_vase")
SETUP = 7  # the GPT-4o descriptions have seven words
OUT = ROOT / "results" / "metrics" / "gpt4o_reproduction.csv"


def main() -> None:
    summary = pd.read_csv(ROOT / "results" / "metrics" / "summary.csv")
    rows = []
    for pair in PAIRS:
        data = arms_mod.load_arm(ROOT, pair, "gpt4o", SETUP)
        keep = np.flatnonzero(arms_mod.nonempty(data))
        stem, cls = data.stem[keep], data.cls[keep]
        neg, pos = sorted(set(cls))
        x = arms_mod.caption_means(data)[keep]
        folds = folds_mod.load_folds(ROOT / "results" / "folds" / f"{pair}.csv")

        accuracy, gap = [], 0.0
        for repeat in range(int(folds["repeat"].max()) + 1):
            assign = folds_mod.fold_of(folds, repeat)
            fold = np.array([assign[s] for s in stem])
            for f in sorted(set(fold)):
                test, train = np.flatnonzero(fold == f), np.flatnonzero(fold != f)
                clf = LogisticRegression(max_iter=2000, class_weight="balanced")
                clf.fit(x[train], cls[train])
                p = clf.predict_proba(x[test])[:, 1]
                s_, y_, p_ = evaluate.by_image(stem[test], cls[test], p)
                accuracy.append(float(np.mean((p_ >= 0.5) == (y_ == pos))))

                # SMER: the sentence logit is the bias plus the mean of the word logits
                z = smer.vocab_logits(data.store, clf.coef_.ravel().astype(np.float32))
                words = smer.explain(data, z, float(clf.intercept_[0]), neg, pos,
                                     rows=keep[test])
                smer_logit = words.groupby(["stem", "rep"], sort=False).apply(
                    lambda g: g["bias"].iloc[0] + g["z"].astype(float).mean(),
                    include_groups=False)
                model_logit = pd.Series(
                    clf.decision_function(x[test]),
                    index=pd.MultiIndex.from_arrays([stem[test], data.rep[keep][test]]))
                gap = max(gap, float(np.abs(
                    smer_logit.values - model_logit.reindex(smer_logit.index).values).max()))

        stored = summary[(summary.pair == pair) & (summary.arm == "gpt4o")
                         & (summary.level == "image")
                         & (summary.metric == "accuracy")].iloc[0]
        rows.append({"pair": pair, "images": len(set(stem)), "fits": len(accuracy),
                     "accuracy_reproduced": float(np.mean(accuracy)),
                     "accuracy_stored": float(stored["mean"]),
                     "difference": float(np.mean(accuracy) - stored["mean"]),
                     "max_logit_gap": gap})
        print(rows[-1], flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

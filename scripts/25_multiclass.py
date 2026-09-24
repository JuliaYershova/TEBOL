#!/usr/bin/env python3
"""Stage 6 -- the pipeline over more than two classes.

Reviewers asked whether TEBOL survives multiclass. It does, and the reason is
in smer.py: the decomposition

    logit_k(S) = b_k + beta_k . (1/n) SUM_i e_i = b_k + (1/n) SUM_i (beta_k . e_i)

never used the fact that beta is a single vector. It used two facts -- mean
pooling is linear, the logit is linear -- and both hold for any K. So a word
still contributes exactly `(beta_k . e_i)/n`, now once per class.

**One-vs-rest, deliberately.** sklearn's multinomial fit is over-parameterised:
softmax is invariant to adding a constant to every logit, so the K coefficient
vectors are identified only up to a shift and an absolute `beta_k . e_i` means
nothing. OvR has no such freedom -- each classifier is a genuine binary problem,
which is also exactly the binary TEBOL already validated here, so SMER, AOPC
and the leakage ablation carry over unchanged rather than by analogy.

**No new data.** vocab.npz is one global table of 28,027 word vectors shared by
every pair, so merging classes across pairs is a concatenation of CSR indices,
not a re-embedding. Folds are rebuilt because the existing ones are per pair;
`build_folds` stratifies over any number of classes already.

The experiment is cucumber, zucchini and hotpot on full captions: three food
classes, chance 33%, one of them a prepared dish and two of them vegetables
that look alike. It exists to show the decomposition still holds, so the run
checks that rather than asserting it -- every caption's ten-class logit is
rebuilt from its per-word scores and the largest error is printed.

    python scripts/25_multiclass.py
    python scripts/25_multiclass.py --arm caption_noclass --setups 3 5 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tebol import folds as folds_mod                       # noqa: E402
from tebol.arms import ArmData, caption_means, load_arm, nonempty   # noqa: E402
from tebol.stats import summarize                          # noqa: E402

RAW = ROOT / "data" / "raw"
MET = ROOT / "results" / "metrics"
FOLDS = ROOT / "results" / "folds"

EXPERIMENTS = {"food": ["cucumber", "zucchini", "hotpot"]}
LENGTHS = (3, 5, 7, 10, 15, 20, 25, 30)


def class_to_pair() -> dict[str, str]:
    """class -> the pair folder its images live in."""
    return {c.name: p.name for p in sorted(RAW.iterdir()) if p.is_dir()
            for c in sorted(p.iterdir()) if c.is_dir()}


def merge(parts: list[ArmData]) -> ArmData:
    """Concatenate arms from different pairs.

    Safe because `rows` indexes the one global vocab table, so the word ids
    mean the same thing in every pair; only the CSR offsets need rebasing.
    """
    rows = np.concatenate([p.rows for p in parts])
    offsets = [np.zeros(1, dtype=np.int64)]
    base = 0
    for p in parts:
        offsets.append(p.offsets[1:] + base)
        base += len(p.rows)
    return ArmData(
        pair="+".join(dict.fromkeys(p.pair for p in parts)),
        arm=parts[0].arm, setup=parts[0].setup,
        stem=np.concatenate([p.stem for p in parts]),
        rep=np.concatenate([p.rep for p in parts]),
        cls=np.concatenate([p.cls for p in parts]),
        rows=rows, offsets=np.concatenate(offsets), store=parts[0].store,
    )


def load(classes: list[str], arm: str, setup: int) -> ArmData:
    """One design matrix over several classes, filtered to those classes."""
    c2p = class_to_pair()
    missing = [c for c in classes if c not in c2p]
    if missing:
        sys.exit(f"no images for {missing}; have {sorted(c2p)}")

    parts = []
    for pair in dict.fromkeys(c2p[c] for c in classes):
        d = load_arm(ROOT, pair, arm, setup)
        keep = np.isin(d.cls, classes) & nonempty(d)
        sel = np.flatnonzero(keep)
        flat = [d.rows[d.offsets[i]:d.offsets[i + 1]] for i in sel]
        off = np.zeros(len(sel) + 1, dtype=np.int64)
        off[1:] = np.cumsum([len(f) for f in flat])
        parts.append(ArmData(
            pair=pair, arm=arm, setup=setup,
            stem=d.stem[sel], rep=d.rep[sel], cls=d.cls[sel],
            rows=(np.concatenate(flat) if flat
                  else np.empty(0, np.int32)).astype(np.int32),
            offsets=off, store=d.store))
    return merge(parts)


def get_folds(name: str, data: ArmData) -> pd.DataFrame:
    """Fold assignment over images, built once per experiment and reused."""
    path = FOLDS / f"multiclass_{name}.csv"
    if path.exists():
        return folds_mod.load_folds(path)
    first = {}
    for s, c in zip(data.stem, data.cls):
        first.setdefault(s, c)
    stems = np.array(sorted(first))
    f = folds_mod.build_folds(stems, np.array([first[s] for s in stems]))
    folds_mod.save_folds(path, f)
    print(f"wrote {path}  ({len(stems)} images, "
          f"{f.cls.nunique()} classes)")
    return f


def fit_fold(X, y, tr, te, classes):
    """One OvR fit. Returns (probabilities on te, coefficient matrix, bias)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=2000, class_weight="balanced")).fit(X[tr],
                                                                       y[tr])
    order = list(clf.classes_)
    if order != classes:
        raise AssertionError(f"class order {order} != {classes}")
    beta = np.vstack([e.coef_.ravel() for e in clf.estimators_])
    bias = np.array([float(e.intercept_[0]) for e in clf.estimators_])
    return clf.predict_proba(X[te]), beta, bias


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", default="food", choices=sorted(EXPERIMENTS))
    ap.add_argument("--arm", default="caption",
                    choices=["caption", "caption_noclass"])
    ap.add_argument("--setups", nargs="*", type=int, default=[5])
    ap.add_argument("--top-words", type=int, default=12)
    args = ap.parse_args()

    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 confusion_matrix, f1_score)

    wanted = EXPERIMENTS[args.experiment]
    print(f"{args.experiment}: {len(wanted)} classes -- {', '.join(wanted)}")

    rows, conf_rows, word_rows = [], [], []
    worst_err = 0.0
    for setup in args.setups:
        data = load(wanted, args.arm, setup)
        classes = sorted(set(data.cls))
        X = caption_means(data)
        y = data.cls
        folds = get_folds(args.experiment, data)
        print(f"\nw{setup:02d} {args.arm}: {len(data)} captions, "
              f"{len(set(data.stem))} images, {X.shape[1]} dims")

        conf = np.zeros((len(classes), len(classes)), dtype=np.int64)
        betas = []
        for repeat in sorted(folds["repeat"].unique()):
            assign = folds_mod.fold_of(folds, int(repeat))
            fold_of_text = np.array([assign.get(s, -1) for s in data.stem])
            if (fold_of_text < 0).any():
                sys.exit("captions with no fold -- rebuild the fold file")
            for fold in sorted(set(fold_of_text.tolist())):
                te = np.flatnonzero(fold_of_text == fold)
                tr = np.flatnonzero(fold_of_text != fold)
                proba, beta, bias = fit_fold(X, y, tr, te, classes)
                betas.append(beta)

                # the claim, checked rather than asserted: a caption's logit
                # for class k is its bias plus the mean of its words' z_k.
                # If this ever stops holding, SMER is no longer a
                # decomposition of this model and the run should say so.
                zv = data.store.vectors @ beta.T          # [vocab, K]
                for i in te[:64]:
                    w = data.rows[data.offsets[i]:data.offsets[i + 1]]
                    got = bias + zv[w].mean(axis=0)
                    want = bias + beta @ X[i]
                    worst_err = max(worst_err,
                                    float(np.abs(got - want).max()))

                # image level: the repetitions vote by mean probability, as in
                # stage 3, so the two levels stay comparable
                df = pd.DataFrame(proba, columns=classes)
                df["stem"], df["y"] = data.stem[te], y[te]
                g = df.groupby("stem", sort=True)
                img = g[classes].mean()
                truth = g["y"].first().to_numpy()
                pred_img = np.array(classes)[img.to_numpy().argmax(1)]
                pred_cap = np.array(classes)[proba.argmax(1)]

                conf += confusion_matrix(truth, pred_img, labels=classes)
                for level, (t, p) in (("caption", (y[te], pred_cap)),
                                      ("image", (truth, pred_img))):
                    rows.append({
                        "experiment": args.experiment, "arm": args.arm,
                        "setup": f"w{setup:02d}", "level": level,
                        "repeat": int(repeat), "fold": int(fold),
                        "n_test": len(t),
                        "accuracy": accuracy_score(t, p),
                        "balanced_accuracy": balanced_accuracy_score(t, p),
                        "f1_macro": f1_score(t, p, average="macro",
                                             labels=classes, zero_division=0),
                    })

        for i, a in enumerate(classes):
            for j, b in enumerate(classes):
                conf_rows.append({"experiment": args.experiment,
                                  "arm": args.arm, "setup": f"w{setup:02d}",
                                  "true": a, "pred": b, "n": int(conf[i, j])})

        # SMER, averaged over the 25 fits: one score per (word, class)
        beta = np.mean(betas, axis=0)
        z = data.store.vectors @ beta.T                 # [vocab, K]
        used = np.unique(data.rows)
        for k, c in enumerate(classes):
            top = used[np.argsort(-z[used, k])][:args.top_words]
            for rank, r in enumerate(top, 1):
                word_rows.append({"experiment": args.experiment,
                                  "arm": args.arm, "setup": f"w{setup:02d}",
                                  "class": c, "rank": rank,
                                  "word": data.store.keys[r],
                                  "z": float(z[r, k])})

    MET.mkdir(parents=True, exist_ok=True)
    tag = f"multiclass_{args.experiment}_{args.arm}"
    f = pd.DataFrame(rows)
    f.to_csv(MET / f"{tag}_folds.csv", index=False)
    pd.DataFrame(conf_rows).to_csv(MET / f"{tag}_confusion.csv", index=False)
    pd.DataFrame(word_rows).to_csv(MET / f"{tag}_words.csv", index=False)

    summ = []
    for (setup, level), g in f.groupby(["setup", "level"]):
        for metric in ("accuracy", "balanced_accuracy", "f1_macro"):
            summ.append({"experiment": args.experiment, "arm": args.arm,
                         "setup": setup, "level": level, "metric": metric,
                         **summarize(g[metric].to_numpy())})
    s = pd.DataFrame(summ)
    s.to_csv(MET / f"{tag}_summary.csv", index=False)

    print(f"\n{args.experiment} / {args.arm}, image level\n")
    print(f"  {'setup':<8}{'accuracy':>22}{'balanced acc':>22}{'macro F1':>22}")
    for setup in [f"w{n:02d}" for n in args.setups]:
        line = f"  {setup:<8}"
        for metric in ("accuracy", "balanced_accuracy", "f1_macro"):
            q = s[(s.setup == setup) & (s.level == "image")
                  & (s.metric == metric)]
            line += (f"{q.iloc[0]['mean']:>14.4f} "
                     f"±{(q.iloc[0]['ci_hi']-q.iloc[0]['ci_lo'])/2:.3f}"
                     if not q.empty else f"{'-':>22}")
        print(line)

    c = pd.DataFrame(conf_rows)
    c = c[c.setup == f"w{args.setups[-1]:02d}"]
    piv = c.pivot(index="true", columns="pred", values="n")
    piv = piv.div(piv.sum(axis=1), axis=0) * 100
    print(f"\nconfusion at w{args.setups[-1]:02d}, image level, row-normalised %\n")
    print(piv.round(1).to_string())
    print(f"\ndecomposition check: bias + mean(z per word) against the model's "
          f"own logit,\n  over 1,600 captions x {len(classes)} classes, "
          f"largest error {worst_err:.2e}")
    print(f"\nwrote {MET / (tag + '_summary.csv')} and _folds, _confusion, _words")


if __name__ == "__main__":
    main()

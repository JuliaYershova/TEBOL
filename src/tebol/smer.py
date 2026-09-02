"""SMER by decomposition: the classifier's logit, split across the words.

The classifier is logistic regression on the *mean* of a caption's word
vectors, and that makes the split exact rather than estimated:

    z(S) = b + w . (1/n) SUM_i e_i = b + (1/n) SUM_i (w . e_i)

so word i contributes exactly `(w . e_i)/n`, and bias plus those contributions
reconstructs the caption logit to floating-point error. Nothing is perturbed,
sampled, or fitted -- which is the whole argument against LIME on this model.

Three quantities come out of it and they are not interchangeable:

    z = w . e            ranking. A property of the word alone, so it needs no
                         context and costs one dot product.
    expit(z + b)         reporting. "What the model would say if this word were
                         the entire caption" -- the number that goes in tables
                         and under bounding boxes. Equal to predict_proba([e]).
    z / n                the decomposition claim. Sums with b to the caption
                         logit; only meaningful inside one caption.

**Leave-one-out gives the same ranking.** Dropping word i moves the mean to
`(n.mu - e_i)/(n - 1)`, so the logit falls by `(z_i - zbar)/(n - 1)`. Within a
caption `zbar` and `n` are constants and expit is monotone, so ordering words
by the perturbation drop and ordering them by `z` produce the identical list --
at n predict_proba calls instead of one matrix-vector product. The perturbation
implementation in the Diplom notebooks is therefore an expensive way to compute
this, not a different method.

The same algebra says something less comfortable that belongs in the write-up:
because `z` depends only on the word, SMER's ranking is *context-free*. The
same word ranks identically in every caption it appears in. That is exactly the
expressiveness LIME has and this does not, and it is a limitation of the model
being linear over a mean, not of the implementation.

**What AOPC needs.** Removing a set of positions leaves
`b + mean(z[kept])` -- so the per-word scalars are sufficient, and the 2560-dim
vectors are not needed again. That is why `explain` writes `z` per word rather
than a probability per word: every AOPC variant, local or global, at any k, is
arithmetic over this one table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

from .arms import ArmData


def vocab_logits(store, beta: np.ndarray) -> np.ndarray:
    """[n_vocab] -- `w . e` for every word in the vocabulary, in one product.

    Computed over the vocabulary rather than per caption because a word's
    vector does not depend on the caption: 10.7k dot products serve 1.7M word
    occurrences.
    """
    return (store.vectors @ np.asarray(beta, dtype=store.vectors.dtype)).astype(np.float64)


def _positions(offsets: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """0,1,2,... within each caption, over the flat CSR."""
    return np.arange(offsets[-1]) - np.repeat(offsets[:-1], lengths)


def explain(data: ArmData, z_vocab: np.ndarray, bias: float,
            neg: str, pos: str, rows: np.ndarray | None = None,
            fold: int = -1, repeat: int = -1) -> pd.DataFrame:
    """One row per word occurrence, for the texts in `rows`.

    `rows` selects which texts to explain -- pass a fold's test indices so the
    coefficients explaining a caption never saw it. `bias` and `z_vocab` must
    come from that same fold's fit.

    Positions are kept because words repeat: 'a red truck and a ladder' has two
    occurrences of 'a', they must be removable independently, and a table keyed
    on the surface form alone cannot do that.
    """
    sel = np.arange(len(data)) if rows is None else np.asarray(rows)
    sel = sel[data.lengths[sel] > 0]

    # CSR restricted to the selected texts
    flat = np.concatenate([data.word_rows(i) for i in sel]) if sel.size \
        else np.empty(0, dtype=np.int32)
    lengths = data.lengths[sel]
    offsets = np.zeros(len(sel) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])

    z = z_vocab[flat]
    sent_logit = bias + np.add.reduceat(z, offsets[:-1]) / lengths
    sent_prob = expit(sent_logit)
    pred = np.where(sent_prob >= 0.5, pos, neg)

    return pd.DataFrame({
        "stem": np.repeat(data.stem[sel], lengths),
        "rep": np.repeat(data.rep[sel], lengths),
        "cls": np.repeat(data.cls[sel], lengths),
        "repeat": np.int16(repeat),
        "fold": np.int16(fold),
        "pos": _positions(offsets, lengths).astype(np.int16),
        "word": [data.store.keys[r] for r in flat],
        "z": z.astype(np.float32),
        "word_prob": expit(z + bias).astype(np.float32),
        "share": (z / np.repeat(lengths, lengths)).astype(np.float32),
        "n_words": np.repeat(lengths, lengths).astype(np.int16),
        "bias": np.float32(bias),
        "sent_logit": np.repeat(sent_logit, lengths).astype(np.float32),
        "sent_prob": np.repeat(sent_prob, lengths).astype(np.float32),
        "pred_class": np.repeat(pred, lengths),
    })


def align(df: pd.DataFrame, target: str, pos: str) -> pd.Series:
    """Per-word probability read against a class of interest.

    `target` is 'pred' for AOPC -- the curve measures the fall in the model's
    own confidence, so it has to follow what the model actually said -- and
    'true' for per-caption display, where the question is how much a word
    supports the correct answer.
    """
    ref = df["pred_class"] if target == "pred" else df["cls"]
    return np.where(ref == pos, df["word_prob"], 1.0 - df["word_prob"])


def global_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Corpus-level word ranking, with its across-fold spread.

    `z` is context-free, so a word has one score per fold and the only
    variation across occurrences is which fold's coefficients produced it.
    `z_sd` is therefore a stability check on the ranking itself, free: a word
    whose score swings between folds is not a finding.
    """
    g = df.groupby("word", sort=False).agg(
        n_occ=("z", "size"),
        n_texts=("stem", "nunique"),
        z_mean=("z", "mean"),
        z_sd=("z", "std"),
        word_prob_mean=("word_prob", "mean"),
    ).reset_index()
    g["z_sd"] = g["z_sd"].fillna(0.0)
    g["abs_z"] = g["z_mean"].abs()
    return g.sort_values("z_mean", ascending=False, ignore_index=True)


def caption_logit(z: np.ndarray, bias: float,
                  drop: set[int] | None = None) -> float:
    """The primitive AOPC is built from: logit after removing some positions.

    Kept here so the claim in the module docstring is executable rather than
    asserted -- given the per-word `z` of one caption, no embedding, no model
    and no vector are needed to score any perturbation of it.
    """
    keep = np.ones(len(z), dtype=bool)
    if drop:
        keep[list(drop)] = False
    return float(bias + z[keep].mean()) if keep.any() else float(bias)

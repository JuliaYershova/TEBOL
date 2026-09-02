"""Scoring one fold, at both the levels the thesis needs.

**Caption level** is one row per generated caption. It is the level the model
is fit at, and the level SMER explains, so it is the honest denominator for
anything about words.

**Image level** means the five repetitions' probabilities and scores once per
photograph. This is the number that answers "can this pipeline classify the
image", which is the claim the thesis actually makes -- five captions of one
vase are five looks at one decision, not five decisions. It is also the level
the bootstrap resamples at, because the reps are not independent draws.

Both levels come out of the same fold, so reporting them side by side costs
nothing and pre-empts the obvious objection either way round.

Calibration metrics -- log loss and Brier -- are not decoration here. SMER
reads `expit(beta . e + b)` as a probability and puts that number in tables and
under bounding boxes. If the classifier is confidently wrong the ranking can
still be fine while every reported probability is misleading, and only a proper
scoring rule shows that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe(fn, *args, **kw) -> float:
    """A fold with one class present makes AUC undefined; report NaN, not a raise."""
    try:
        return float(fn(*args, **kw))
    except ValueError:
        return float("nan")


def score(y_true: np.ndarray, p_pos: np.ndarray, neg: str, pos: str) -> dict:
    """Every metric for one set of predictions.

    `p_pos` is P(pos) -- column 1 of `predict_proba`, matching `classes_[1]`.
    """
    y_true = np.asarray(y_true)
    p_pos = np.asarray(p_pos, dtype=float)
    y_pred = np.where(p_pos >= 0.5, pos, neg)
    y_bin = (y_true == pos).astype(int)
    labels = [neg, pos]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    (tn, fp), (fn, tp) = cm

    out = {
        "n": len(y_true),
        "n_pos": int(y_bin.sum()),
        "accuracy": _safe(accuracy_score, y_true, y_pred),
        "balanced_accuracy": _safe(balanced_accuracy_score, y_true, y_pred),
        "precision_macro": _safe(precision_score, y_true, y_pred,
                                 labels=labels, average="macro", zero_division=0),
        "recall_macro": _safe(recall_score, y_true, y_pred,
                              labels=labels, average="macro", zero_division=0),
        "f1_macro": _safe(f1_score, y_true, y_pred,
                          labels=labels, average="macro", zero_division=0),
        "roc_auc": _safe(roc_auc_score, y_bin, p_pos),
        "pr_auc": _safe(average_precision_score, y_bin, p_pos),
        "log_loss": _safe(log_loss, y_bin, p_pos, labels=[0, 1]),
        "brier": _safe(brier_score_loss, y_bin, p_pos),
        "mcc": _safe(matthews_corrcoef, y_true, y_pred),
        "kappa": _safe(cohen_kappa_score, y_true, y_pred, labels=labels),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    for cls, avg in ((neg, 0), (pos, 1)):
        mask_lbl = [cls]
        out[f"precision_{cls}"] = _safe(precision_score, y_true, y_pred,
                                        labels=mask_lbl, average="macro",
                                        zero_division=0)
        out[f"recall_{cls}"] = _safe(recall_score, y_true, y_pred,
                                     labels=mask_lbl, average="macro",
                                     zero_division=0)
        out[f"f1_{cls}"] = _safe(f1_score, y_true, y_pred, labels=mask_lbl,
                                 average="macro", zero_division=0)
    return out


def by_image(stem: np.ndarray, y_true: np.ndarray,
             p_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse a fold's captions to one prediction per image, by mean probability.

    Mean of probabilities rather than a vote: it keeps the metric proper, so
    log loss and Brier stay meaningful at this level too, and it does not throw
    away the margin on a 3-2 split.
    """
    df = pd.DataFrame({"stem": stem, "y": y_true, "p": p_pos})
    g = df.groupby("stem", sort=True).agg(y=("y", "first"), p=("p", "mean"))
    return g.index.to_numpy(), g["y"].to_numpy(), g["p"].to_numpy()

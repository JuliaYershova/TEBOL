"""Confidence intervals and arm-to-arm tests, both corrected for the design.

Two corrections matter here and both are easy to skip by accident.

**Resample images, not captions.** Each image contributes five captions of the
same photograph. Bootstrapping over captions would treat five correlated rows
as five independent draws and return an interval roughly sqrt(5) too narrow.
`bootstrap_ci` resamples stems and takes all of a stem's rows with it.

**Cross-validation folds are not independent samples.** Any two training sets
in 5-fold CV share 3/5 of their rows, so the usual t-interval over fold scores
understates the variance and declares differences significant that a rerun
would not reproduce. `nadeau_bengio` applies the standard correction, which
inflates the variance by `1/k + n_test/n_train` instead of `1/k`.

Which test to reach for: `nadeau_bengio` for the headline arm comparisons, and
`wilcoxon` when the per-fold differences are visibly skewed or the reviewer
prefers a distribution-free statement -- the survey chapter already uses
Wilcoxon, so the thesis stays internally consistent either way. Both are paired
and both require the arms to have been run on the same folds, which
`tebol.folds` guarantees by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def bootstrap_ci(stem: np.ndarray, y_true: np.ndarray, p_pos: np.ndarray,
                 metric, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 20260824) -> tuple[float, float]:
    """Percentile interval for `metric(y, p)`, resampling whole images.

    `metric` takes (y_true, p_pos) and returns a float. Draws that lose a class
    entirely give NaN for AUC-like metrics and are dropped rather than counted.
    """
    df = pd.DataFrame({"stem": stem, "y": y_true, "p": p_pos})
    groups = list(df.groupby("stem", sort=True).indices.values())
    rng = np.random.default_rng(seed)
    y = df["y"].to_numpy()
    p = df["p"].to_numpy()

    vals = np.empty(n_boot)
    n = len(groups)
    for b in range(n_boot):
        pick = rng.integers(0, n, n)
        idx = np.concatenate([groups[i] for i in pick])
        vals[b] = metric(y[idx], p[idx])

    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    return (float(np.quantile(vals, alpha / 2)),
            float(np.quantile(vals, 1 - alpha / 2)))


def nadeau_bengio(diffs: np.ndarray, n_train: int,
                  n_test: int) -> tuple[float, float]:
    """Corrected resampled t-test over per-fold differences.

    `diffs` is one paired difference per fold (across every repeat). Returns
    (t, two-sided p). The correction is the `n_test/n_train` term; without it
    this is the ordinary paired t-test and is anti-conservative.
    """
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    k = len(d)
    if k < 2:
        return float("nan"), float("nan")
    var = d.var(ddof=1)
    if var == 0:
        return (float("inf") if d.mean() else 0.0), (0.0 if d.mean() else 1.0)
    t = d.mean() / np.sqrt(var * (1.0 / k + n_test / n_train))
    return float(t), float(2 * stats.t.sf(abs(t), df=k - 1))


def wilcoxon(diffs: np.ndarray) -> tuple[float, float]:
    """Distribution-free paired alternative, over the same per-fold differences."""
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 2 or np.all(d == 0):
        return float("nan"), float("nan")
    res = stats.wilcoxon(d)
    return float(res.statistic), float(res.pvalue)


def bonferroni(p: np.ndarray) -> np.ndarray:
    """Family-wise correction, matching the survey chapter's convention."""
    p = np.asarray(p, dtype=float)
    return np.minimum(p * np.isfinite(p).sum(), 1.0)


def summarize(values: np.ndarray, alpha: float = 0.05) -> dict:
    """mean / sd / normal-theory interval over fold scores.

    Reported alongside the bootstrap interval, not instead of it: this one
    describes the spread *between folds*, which is what a reader comparing two
    arms wants, while the bootstrap describes sampling error in the images.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"mean": float("nan"), "sd": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"), "k": 0}
    if v.size == 1:
        return {"mean": float(v[0]), "sd": 0.0,
                "ci_lo": float(v[0]), "ci_hi": float(v[0]), "k": 1}
    se = v.std(ddof=1) / np.sqrt(v.size)
    h = se * stats.t.ppf(1 - alpha / 2, df=v.size - 1)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
            "ci_lo": float(v.mean() - h), "ci_hi": float(v.mean() + h),
            "k": int(v.size)}

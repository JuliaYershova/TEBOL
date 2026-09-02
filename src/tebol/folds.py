"""Cross-validation folds, assigned to images and shared by every arm.

Two facts about the data decide the whole design here.

**An image owns 20 captions.** Stage 1 wrote four lengths (3/5/7/10 words) at
five repetitions each. Splitting on captions would put reps 0-3 of an image in
train and rep 4 in test, and since the five reps describe the same photograph
the test score would measure memorisation. Folds are therefore assigned to
*images*, once, and captions inherit the fold of their stem.

**Arms have to be comparable.** The five arms answer one question each --
does removing the class word hurt, do tags beat descriptions -- and every one
of those is a *difference* between two arms. A difference is only interpretable
if both sides saw the same images in the same partition, so the assignment is
computed once per pair and reused by all arms and all setups. The tag arms are
this same table filtered to the images ImageNet-Captions covers, which keeps
them paired with the caption arms at image level.

Repeated stratified k-fold, 5 x 5. Stratified because the pairs are unbalanced
(hotpot/vase is 203/270 once tags are required); repeated because five folds
give five numbers and a confidence interval off five numbers is mostly noise.
Twenty-five fits per arm is cheap here -- the design matrix is at most 12k x
2560 and logistic regression on it takes seconds.

Written to results/folds/<pair>.csv, which is tracked in git: the fold
assignment is part of what makes a reported number reproducible, and it is a
few hundred kilobytes of integers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

#: Fixed so a rerun reproduces the split. Changing it invalidates every metric
#: already in results/, which is why it lives here and not in a CLI flag.
SEED = 20260824

N_SPLITS = 5
N_REPEATS = 5


def build_folds(stems: np.ndarray, classes: np.ndarray,
                n_splits: int = N_SPLITS, n_repeats: int = N_REPEATS,
                seed: int = SEED) -> pd.DataFrame:
    """Assign every image a fold, once per repeat.

    `stems` and `classes` are parallel arrays over *distinct images*. Returns
    long rows of (stem, cls, repeat, fold); an image appears `n_repeats` times,
    once per repeat, and is in the test set exactly once within each repeat.
    """
    stems = np.asarray(stems)
    classes = np.asarray(classes)
    if len(stems) != len(classes):
        raise ValueError(f"{len(stems)} stems but {len(classes)} classes")
    if len(set(stems)) != len(stems):
        raise ValueError("stems must be distinct -- pass images, not captions")

    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=seed)
    fold = np.empty((n_repeats, len(stems)), dtype=np.int16)
    for i, (_, test) in enumerate(cv.split(stems, classes)):
        fold[i // n_splits, test] = i % n_splits

    return pd.DataFrame({
        "stem": np.tile(stems, n_repeats),
        "cls": np.tile(classes, n_repeats),
        "repeat": np.repeat(np.arange(n_repeats, dtype=np.int16), len(stems)),
        "fold": fold.ravel(),
    })


def save_folds(path: Path, folds: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(path, index=False)


def load_folds(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"stem": str, "cls": str})


def fold_of(folds: pd.DataFrame, repeat: int) -> dict[str, int]:
    """{stem: fold} for one repeat -- the lookup captions are scored through."""
    sub = folds[folds["repeat"] == repeat]
    return dict(zip(sub["stem"], sub["fold"].astype(int)))

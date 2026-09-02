"""The decomposition claims, checked against sklearn rather than asserted.

Stage 3 replaces the Diplom notebooks' leave-one-out loop with a closed form,
and everything downstream -- the word rankings, the AOPC table, the bounding
boxes -- inherits whatever that form gets wrong. These are the four identities
it rests on. They are cheap and they fail loudly if the mean-pooling, the class
order or the coefficient extraction ever drifts.

Run with the repo's venv:

    .venv/bin/python -m pytest tests/test_smer_decomposition.py -v

Tolerances are 1e-5, not exact: `beta` is stored float32 to keep 25 folds x
2560 dims small, and the reconstruction accumulates rounding across a caption's
words. The identities are exact in real arithmetic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tebol import arms as arms_mod  # noqa: E402
from tebol import smer  # noqa: E402

PAIR, ARM, SETUP = "hotpot_vase", "caption", 7
TOL = 1e-5


@pytest.fixture(scope="module")
def fitted():
    emb = ROOT / "artifacts" / "embeddings" / arms_mod.EMB_FULL
    if not (emb / "vocab.npz").exists():
        pytest.skip("stage 2 embeddings not present")
    data = arms_mod.load_arm(ROOT, PAIR, ARM, SETUP)
    X = arms_mod.caption_means(data)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X, data.cls)
    beta = clf.coef_.ravel().astype(np.float32)
    bias = float(clf.intercept_[0])
    return data, X, clf, beta, bias, smer.vocab_logits(data.store, beta)


def test_sentence_logit_matches_sklearn(fitted):
    """b + mean(w.e_i) is the model's own decision function, not an approximation."""
    data, X, clf, _, bias, z_vocab = fitted
    idx = range(0, len(data), 997)
    mine = np.array([bias + z_vocab[data.word_rows(i)].mean() for i in idx])
    assert np.allclose(mine, clf.decision_function(X[list(idx)]), atol=TOL)


def test_shares_reconstruct_the_logit(fitted):
    """b + SUM_i (w.e_i)/n == z(S): the attribution is complete."""
    data, X, clf, _, bias, z_vocab = fitted
    for i in range(0, len(data), 1499):
        z = z_vocab[data.word_rows(i)]
        assert bias + (z / len(z)).sum() == pytest.approx(
            float(clf.decision_function(X[i:i + 1])[0]), abs=TOL)


def test_word_probability_is_that_word_alone(fitted):
    """expit(w.e + b) is what the model would say given only that word."""
    data, _, clf, _, bias, z_vocab = fitted
    rows = data.word_rows(12)
    assert np.allclose(expit(z_vocab[rows] + bias),
                       clf.predict_proba(data.store.vectors[rows])[:, 1], atol=TOL)


def test_leave_one_out_gives_the_same_ranking(fitted):
    """The expensive perturbation loop and the dot product order words alike.

    This is what licenses replacing the Diplom implementation: within a caption
    the LOO drop is (z_i - zbar)/(n - 1), an increasing affine function of z_i,
    so the two rankings cannot disagree.
    """
    data, _, clf, _, _, z_vocab = fitted
    checked = 0
    for i in range(0, len(data), 53):
        rows = data.word_rows(i)
        if len(rows) < 3:
            continue
        E = data.store.vectors[rows]
        full = clf.predict_proba([E.mean(0)])[0, 1]
        loo = [full - clf.predict_proba([np.delete(E, k, axis=0).mean(0)])[0, 1]
               for k in range(len(rows))]
        assert (np.argsort(np.argsort(loo)) ==
                np.argsort(np.argsort(z_vocab[rows]))).all(), f"caption {i}"
        checked += 1
        if checked >= 40:
            break
    assert checked >= 10


def test_aopc_primitive_needs_only_the_scalars(fitted):
    """Removing words is arithmetic on z -- no vector, no model, no re-embed."""
    data, _, clf, _, bias, z_vocab = fitted
    rows = data.word_rows(7)
    z = z_vocab[rows]
    drop = {0, 2}
    keep = [k for k in range(len(rows)) if k not in drop]
    direct = float(clf.decision_function([data.store.vectors[rows[keep]].mean(0)])[0])
    assert smer.caption_logit(z, bias, drop) == pytest.approx(direct, abs=TOL)

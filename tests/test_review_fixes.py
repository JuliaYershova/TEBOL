"""Regression tests for the code-review findings on LIME, stats and caption IO."""
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tebol import stats  # noqa: E402
from tebol.captions_io import iter_raw  # noqa: E402


def test_lime_bow_weights_follow_words_not_vocabulary_ids():
    from tebol.lime_runner import explain_one
    words = "the red red blue-and-white car on the road".split()
    z = np.array([0, 5, 5, -3, 2, 0, 0, 1.0])
    w = explain_one(words, z, 0.0, True, 3000, 1, bow=True)
    assert w[1] == w[2] != 0.0            # both copies of "red" share a weight
    assert w[3] < 0 < w[1]                # hyphenated token found, sign kept
    assert w[0] == w[6]                   # both copies of "the"
    assert np.argmax(w) in (1, 2)


def test_nadeau_bengio_ci_agrees_with_p_value():
    rng = np.random.default_rng(0)
    for shift in (0.0, 0.004, 0.02):
        d = rng.normal(shift, 0.01, 25)
        _, p, h = stats.nadeau_bengio(d, 800, 200)
        excludes_zero = abs(d.mean()) > h
        assert excludes_zero == (p < 0.05)


def test_iter_raw_falls_back_to_gz(tmp_path):
    rows = [{"stem": "a", "n_words": 5, "rep": 0, "caption": "x"}]
    with gzip.open(tmp_path / "c.jsonl.gz", "wt", encoding="utf-8") as fh:
        fh.write("\n".join(json.dumps(r) for r in rows) + "\n")
    assert list(iter_raw(tmp_path / "c.jsonl")) == rows

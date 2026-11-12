"""The published survey table is anonymous and gives the paper's user-study numbers."""

from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RATINGS = ROOT / "data" / "survey" / "ratings.csv"


def _ratings() -> pd.DataFrame:
    return pd.read_csv(RATINGS)


def test_no_identifying_columns():
    allowed = {"participant", "language", "expertise", "attention_ok"}
    for c in _ratings().columns:
        assert c in allowed or c.startswith("comprehension_") or c.endswith(("_bb", "_ht")), c


def test_ratings_are_one_to_five_and_complete():
    d = _ratings()
    rating = [c for c in d.columns if c.endswith(("_bb", "_ht")) or c.startswith("comprehension_")]
    assert len(d) == 270 and len(rating) == 22
    assert d[rating].isin([1, 2, 3, 4, 5]).all().all()


def test_comprehension_matches_the_paper():
    d = _ratings()
    assert round(d["comprehension_bb"].mean(), 2) == 4.06
    assert round(d["comprehension_ht"].mean(), 2) == 3.49
    assert stats.wilcoxon(d["comprehension_bb"], d["comprehension_ht"]).pvalue < 1e-15

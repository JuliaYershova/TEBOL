#!/usr/bin/env python3
"""Cut the LimeSurvey export down to the answers the paper's statistics use.

The export is not public: it holds a respondent id, timestamps, the recruitment
group, a seed and free-text comments. This script keeps only

    participant         row of the export (0-based); the same key as `response_row`
                        in data/manifest/survey_comment_codes.csv
    language            the language the participant chose (cs, en, uk)
    expertise           self-reported machine-learning experience, in three levels
                        (the two highest levels are merged: 26 people)
    attention_ok        True if all three attention-check questions were answered
                        as instructed
    comprehension_bb / comprehension_ht
                        final question, 1 (Very difficult) ... 5 (Very easy)
    <class>_bb / <class>_ht
                        how well the display captures the named object,
                        1 (Very Poor) ... 5 (Excellent), ten classes per format

and writes data/survey/ratings.csv. Nothing else is copied, so the file
contains no identifier, date, time, group or text. `survey_stats.py` computes every
number of the user study from that file alone.

It also checks that the coded comments (data/manifest/survey_comment_codes.csv)
belong to non-empty comments of the export, which is the one check that needs the
private text.

    python scripts/survey_anonymize.py --data "<path to results-survey214991 (5).csv>"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CODES = ROOT / "data" / "manifest" / "survey_comment_codes.csv"
OUT = ROOT / "data" / "survey" / "ratings.csv"

COMPREHENSION = {"Very easy to comprehend": 5, "Easy to comprehend": 4,
                 "Comprehensible": 3, "Difficult to comprehend": 2,
                 "Very difficult to comprehend": 1}
CAPTURE = {"Excellent": 5, "Good": 4, "Fair": 3, "Poor": 2, "Very Poor": 1}
CLASSES = ["AMBULANCE", "FIRETRUCK", "ANT", "BEE", "ACOUSTICGUITAR", "VIOLIN",
           "HOTPOT", "VASE", "CUCUMBER", "ZUCCHINI"]
EXPERTISE = {"No experience": "none", "Beginner": "beginner",
             "Intermediate": "intermediate or advanced",
             "Advanced": "intermediate or advanced"}
#: the three attention-check questions and the answer each one asks for
ATTENTION = {"G24Q25": "Blue", "G24Q25Copy": "White", "G24Q25CopyCopy": "Yellow"}
#: the final comprehensibility question for each format
COMPREHENSION_COLS = {"bb": "G05Q05", "ht": "G05Q07"}
#: the free-text question for each format
COMMENT_COLS = {"bb": "G29Q31", "ht": "G29Q33"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    args = ap.parse_args()

    d = pd.read_csv(args.data, low_memory=False)
    d = d[d["submitdate"].notna()].reset_index(drop=True)

    out = pd.DataFrame({"participant": d.index})
    out["language"] = d["G28Q29"]
    out["expertise"] = d["G29Q30"].str.split(r"\s*[–-]", n=1, regex=True).str[0] \
        .str.strip().map(EXPERTISE)
    out["attention_ok"] = True
    for col, answer in ATTENTION.items():
        out["attention_ok"] &= (d[col] == answer)
    for fmt, col in COMPREHENSION_COLS.items():
        out[f"comprehension_{fmt}"] = d[col].map(COMPREHENSION)
    for fmt in ("bb", "ht"):
        for c in CLASSES:
            out[f"{c.lower()}_{fmt}"] = d[f"{c}{fmt}"].map(CAPTURE)

    assert not out.isna().any().any(), out.isna().sum()[out.isna().sum() > 0]
    rating_cols = [c for c in out.columns if c.endswith(("_bb", "_ht"))]
    assert out[rating_cols].isin([1, 2, 3, 4, 5]).all().all()

    codes = pd.read_csv(CODES, keep_default_na=False)
    for fmt, col in COMMENT_COLS.items():
        text = d[col].fillna("").astype(str).str.strip()
        nonempty = set(text.index[text != ""])
        c = codes[codes["format"] == fmt]
        assert set(c["response_row"]) == nonempty and c["response_row"].is_unique, \
            f"codes do not match the non-empty {fmt} comments"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(out)} participants, "
          f"{len(out.columns)} columns; {int((~out['attention_ok']).sum())} "
          f"failed an attention check")


if __name__ == "__main__":
    main()

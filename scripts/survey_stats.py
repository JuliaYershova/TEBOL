#!/usr/bin/env python3
"""Statistics of the user study that compares bounding boxes with Grad-CAM heatmaps.

Reads the LimeSurvey export (one row per participant; the file is not part of the
repository, it holds individual responses and free-text comments) and writes only
aggregates:

    results/metrics/survey_stats.csv      every test and mean the paper reports
    results/metrics/survey_counts.csv     the overall comprehensibility answers, per format

Two questions were asked, and they are analysed separately.

    comprehension   one rating per participant at the end, per format:
                    Very difficult = 1 ... Very easy = 5       (columns G05Q05, G05Q07)
    capture         one rating per image, 10 per format, how well the display
                    captures the named object: Very Poor = 1 ... Excellent = 5
                                                              (columns <CLASS>bb, <CLASS>ht)

Each participant rated one bounding-box image and one heatmap for every class,
so every comparison is paired within the participant.

Free-text comments. Each comment was read and coded once by a language model, one
sentiment towards the format per comment (positive, negative, mixed, neutral) and
any of five themes: C clarity or ease, A accuracy of the display against the object,
B boundaries or shape of the marked region, O the image hidden under the overlay,
K colours, gradient or missing legend. The codes (no comment text) are in
data/manifest/survey_comment_codes.csv, keyed by the row of the export; this script
checks that every code matches a non-empty comment and counts them.

    python scripts/survey_stats.py --data "<path to results-survey214991 (5).csv>"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
CODES = ROOT / "data" / "manifest" / "survey_comment_codes.csv"
DATA = Path("/Users/kjzg289/Library/CloudStorage/OneDrive-AZCollaboration/"
            "study/2 semestr/Diplom/data/results-survey214991 (5).csv")
OUT = ROOT / "results" / "metrics"

COMPREHENSION = {"Very easy to comprehend": 5, "Easy to comprehend": 4,
                 "Comprehensible": 3, "Difficult to comprehend": 2,
                 "Very difficult to comprehend": 1}
CAPTURE = {"Excellent": 5, "Good": 4, "Fair": 3, "Poor": 2, "Very Poor": 1}
CLASSES = ["AMBULANCE", "FIRETRUCK", "ANT", "BEE", "ACOUSTICGUITAR", "VIOLIN",
           "HOTPOT", "VASE", "CUCUMBER", "ZUCCHINI"]
#: the final comprehensibility question for each format
COMPREHENSION_COLS = {"bb": "G05Q05", "ht": "G05Q07"}


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    """Wilcoxon signed-rank (zeros dropped) and paired t on the same pairs."""
    w = stats.wilcoxon(a, b)
    t = stats.ttest_rel(a, b)
    return {"n": len(a), "mean_bb": float(a.mean()), "mean_ht": float(b.mean()),
            "wilcoxon_W": float(w.statistic), "wilcoxon_p": float(w.pvalue),
            "t": float(t.statistic), "t_p": float(t.pvalue)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DATA)
    args = ap.parse_args()

    d = pd.read_csv(args.data, low_memory=False)
    d = d[d["submitdate"].notna()].reset_index(drop=True)
    n = len(d)
    rows, counts = [], []

    # --- overall comprehensibility, one rating per participant and format
    comp = {k: d[c].map(COMPREHENSION) for k, c in COMPREHENSION_COLS.items()}
    unmapped = {k: int(v.isna().sum()) for k, v in comp.items()}
    assert not any(unmapped.values()), f"unmapped answers: {unmapped}"
    for fmt, col in COMPREHENSION_COLS.items():
        for answer, score in COMPREHENSION.items():
            counts.append({"format": fmt, "answer": answer, "score": score,
                           "n": int((d[col] == answer).sum())})
    rows.append({"analysis": "comprehension", "group": "all", **paired(
        comp["bb"].to_numpy(float), comp["ht"].to_numpy(float))})

    # --- how well the display captures the named object, per image
    cap = {f: pd.DataFrame({c: d[f"{c}{f}"].map(CAPTURE) for c in CLASSES})
           for f in ("bb", "ht")}
    assert not cap["bb"].isna().any().any() and not cap["ht"].isna().any().any()

    # pooled over every rating (n participants x 10 classes per format)
    pooled_bb, pooled_ht = cap["bb"].to_numpy(float), cap["ht"].to_numpy(float)
    rows.append({"analysis": "capture", "group": "pooled, per rating",
                 **paired(pooled_bb.ravel(), pooled_ht.ravel())})
    # the same, one mean per participant
    rows.append({"analysis": "capture", "group": "pooled, per participant",
                 **paired(pooled_bb.mean(1), pooled_ht.mean(1))})

    per_class = []
    for c in CLASSES:
        r = {"analysis": "capture", "group": c.lower(),
             **paired(cap["bb"][c].to_numpy(float), cap["ht"][c].to_numpy(float))}
        per_class.append(r)
    for r in per_class:
        r["wilcoxon_p_bonferroni"] = min(1.0, r["wilcoxon_p"] * len(CLASSES))
        r["significant"] = r["wilcoxon_p_bonferroni"] < 0.05
        r["better"] = ("bb" if r["mean_bb"] > r["mean_ht"] else "ht")
    rows.extend(per_class)

    # --- who took part
    for col, name in (("startlanguage", "language"), ("G29Q30", "expertise")):
        for value, k in d[col].value_counts().items():
            rows.append({"analysis": f"participants: {name}", "group": str(value)[:40],
                         "n": int(k)})
    rows.append({"analysis": "participants", "group": "submitted", "n": n})
    # --- coded free-text comments
    codes = pd.read_csv(CODES, keep_default_na=False)
    for fmt, col in (("bb", "G29Q31"), ("ht", "G29Q33")):
        text = d[col].fillna("").astype(str).str.strip()
        nonempty = set(text.index[text != ""])
        c = codes[codes["format"] == fmt]
        assert set(c["response_row"]) == nonempty and c["response_row"].is_unique, \
            f"codes do not match the non-empty {fmt} comments"
        rows.append({"analysis": "comments", "group": f"{fmt}: non-empty",
                     "n": len(c)})
        for sent in ("positive", "negative", "mixed", "neutral"):
            rows.append({"analysis": "comment sentiment", "group": f"{fmt}: {sent}",
                         "n": int((c["sentiment"] == sent).sum())})
        for theme, name in (("C", "clarity"), ("A", "accuracy"), ("B", "boundaries"),
                            ("O", "image hidden"), ("K", "colours")):
            has = c["themes"].str.contains(theme)
            rows.append({"analysis": "comment theme", "group": f"{fmt}: {name}",
                         "n": int(has.sum())})

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "survey_stats.csv", index=False)
    pd.DataFrame(counts).to_csv(OUT / "survey_counts.csv", index=False)

    print(f"{n} submitted responses from {args.data.name}")
    show = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(show[show.analysis.isin(["comprehension", "capture"])]
          .drop(columns=["analysis"]).round(4).to_string(index=False))
    print(pd.DataFrame(counts).pivot(index="answer", columns="format",
                                     values="n").to_string())
    print(f"wrote {OUT / 'survey_stats.csv'} and {OUT / 'survey_counts.csv'}")


if __name__ == "__main__":
    main()

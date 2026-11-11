#!/usr/bin/env python3
"""Statistics of the user study that compares bounding boxes with Grad-CAM heatmaps.

Reads data/survey/ratings.csv (one row per participant, no identifiers; made from
the private LimeSurvey export by scripts/survey_anonymize.py) and the coded comments
data/manifest/survey_comment_codes.csv, and writes only aggregates:

    results/metrics/survey_stats.csv      every test and mean the paper reports
    results/metrics/survey_counts.csv     the overall comprehensibility answers, per format

Two questions were asked, and they are analysed separately.

    comprehension   one rating per participant at the end, per format:
                    Very difficult = 1 ... Very easy = 5       (comprehension_bb, _ht)
    capture         one rating per image, 10 per format, how well the display
                    captures the named object: Very Poor = 1 ... Excellent = 5
                                                              (<class>_bb, <class>_ht)

Each participant rated one bounding-box image and one heatmap for every class,
so every comparison is paired within the participant.

Free-text comments. Each comment was coded once by ChatGPT, one sentiment towards
the format per comment (positive, negative, mixed, neutral) and any of five themes:
C clarity or ease, A accuracy of the display against the object, B boundaries or
shape of the marked region, O the image hidden under the overlay, K colours,
gradient or missing legend. The codes (no comment text) are in
data/manifest/survey_comment_codes.csv, keyed by the participant row; this script
counts them.

Sensitivity rows. The preregistration excludes participants who fail the attention
check. The paper's numbers use all submitted responses; the rows
"sensitivity: attention check passed" repeat the two comparisons without the
participants who failed.

    python scripts/survey_stats.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
CODES = ROOT / "data" / "manifest" / "survey_comment_codes.csv"
RATINGS = ROOT / "data" / "survey" / "ratings.csv"
OUT = ROOT / "results" / "metrics"

CLASSES = ["ambulance", "firetruck", "ant", "bee", "acousticguitar", "violin",
           "hotpot", "vase", "cucumber", "zucchini"]
COMPREHENSION = {5: "Very easy to comprehend", 4: "Easy to comprehend",
                 3: "Comprehensible", 2: "Difficult to comprehend",
                 1: "Very difficult to comprehend"}


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
    ap.add_argument("--ratings", type=Path, default=RATINGS)
    args = ap.parse_args()

    d = pd.read_csv(args.ratings)
    n = len(d)
    rows, counts = [], []

    # --- overall comprehensibility, one rating per participant and format
    for fmt in ("bb", "ht"):
        for score, answer in COMPREHENSION.items():
            counts.append({"format": fmt, "answer": answer, "score": score,
                           "n": int((d[f"comprehension_{fmt}"] == score).sum())})
    rows.append({"analysis": "comprehension", "group": "all", **paired(
        d["comprehension_bb"].to_numpy(float), d["comprehension_ht"].to_numpy(float))})

    # --- how well the display captures the named object, per image
    cap = {f: d[[f"{c}_{f}" for c in CLASSES]].set_axis(CLASSES, axis=1)
           for f in ("bb", "ht")}

    # pooled over every rating (n participants x 10 classes per format)
    pooled_bb, pooled_ht = cap["bb"].to_numpy(float), cap["ht"].to_numpy(float)
    rows.append({"analysis": "capture", "group": "pooled, per rating",
                 **paired(pooled_bb.ravel(), pooled_ht.ravel())})
    # the same, one mean per participant
    rows.append({"analysis": "capture", "group": "pooled, per participant",
                 **paired(pooled_bb.mean(1), pooled_ht.mean(1))})

    per_class = []
    for c in CLASSES:
        r = {"analysis": "capture", "group": c,
             **paired(cap["bb"][c].to_numpy(float), cap["ht"][c].to_numpy(float))}
        per_class.append(r)
    for r in per_class:
        r["wilcoxon_p_bonferroni"] = min(1.0, r["wilcoxon_p"] * len(CLASSES))
        r["significant"] = r["wilcoxon_p_bonferroni"] < 0.05
        r["better"] = ("bb" if r["mean_bb"] > r["mean_ht"] else "ht")
    rows.extend(per_class)

    # --- who took part
    for col, name in (("language", "language"), ("expertise", "expertise")):
        for value, k in d[col].value_counts().items():
            rows.append({"analysis": f"participants: {name}", "group": str(value),
                         "n": int(k)})
    rows.append({"analysis": "participants", "group": "submitted", "n": n})
    rows.append({"analysis": "participants", "group": "failed an attention check",
                 "n": int((~d["attention_ok"]).sum())})

    # --- coded free-text comments
    codes = pd.read_csv(CODES, keep_default_na=False)
    assert codes["response_row"].isin(d["participant"]).all()
    for fmt in ("bb", "ht"):
        c = codes[codes["format"] == fmt]
        assert c["response_row"].is_unique
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

    # --- sensitivity: without the participants who failed an attention check
    ok = d[d["attention_ok"]]
    group = "attention check passed"
    rows.append({"analysis": "sensitivity: comprehension", "group": group, **paired(
        ok["comprehension_bb"].to_numpy(float), ok["comprehension_ht"].to_numpy(float))})
    okbb = ok[[f"{c}_bb" for c in CLASSES]].to_numpy(float)
    okht = ok[[f"{c}_ht" for c in CLASSES]].to_numpy(float)
    rows.append({"analysis": "sensitivity: capture", "group": f"{group}, pooled, per rating",
                 **paired(okbb.ravel(), okht.ravel())})

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "survey_stats.csv", index=False)
    pd.DataFrame(counts).to_csv(OUT / "survey_counts.csv", index=False)

    print(f"{n} participants from {args.ratings.name}")
    show = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(show[show.analysis.isin(["comprehension", "capture"])]
          .drop(columns=["analysis"]).round(4).to_string(index=False))
    print(pd.DataFrame(counts).pivot(index="answer", columns="format",
                                     values="n").to_string())
    print(f"wrote {OUT / 'survey_stats.csv'} and {OUT / 'survey_counts.csv'}")


if __name__ == "__main__":
    main()

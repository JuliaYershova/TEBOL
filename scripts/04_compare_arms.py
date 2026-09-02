#!/usr/bin/env python3
"""Stage 3b -- the arm-to-arm tests the five setups were built to support.

Reads results/metrics/stage3/folds.csv and writes comparisons.csv. Nothing is
refitted; the arms already ran on identical image folds, so every comparison
here is paired fold by fold, which is the only reason the differences are
testable at all.

Five families, one per question a reviewer will ask:

    leakage        caption            vs caption_noclass
                   how much of the accuracy was the caption naming its class

    tags_vs_desc   tags               vs caption_tagsub
                   the reviewers' question, on the images tags cover

    tags_vs_clean  tags               vs caption_noclass_tagsub
                   the same question with the leak closed on the caption side

    subset_cost    caption            vs caption_tagsub
                   what restricting to tag-covered images costs on its own,
                   so the two comparisons above can be read against a baseline

    length         w03/w05/w07/w10 against each other, within an arm
                   whether asking the captioner for more words helps

Both tests are reported. Nadeau-Bengio is the headline: the naive paired t over
CV folds ignores that the training sets overlap and will call differences
significant that a rerun would not reproduce. Wilcoxon is carried alongside
because the survey chapter uses it and a reviewer may prefer a distribution-free
statement. Bonferroni is applied within each family, matching the survey's
convention.

    python scripts/04_compare_arms.py
    python scripts/04_compare_arms.py --metric f1_macro --level image
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tebol import stats  # noqa: E402

METRICS_DIR = ROOT / "results" / "metrics" / "stage3"

#: (family, arm_a, arm_b) -- a positive delta always means arm_a scored higher.
ARM_FAMILIES = [
    ("leakage", "caption", "caption_noclass"),
    ("leakage", "caption_tagsub", "caption_noclass_tagsub"),
    ("tags_vs_desc", "caption_tagsub", "tags"),
    ("tags_vs_clean", "caption_noclass_tagsub", "tags"),
    ("subset_cost", "caption", "caption_tagsub"),
]

DEFAULT_METRICS = ("accuracy", "balanced_accuracy", "f1_macro", "roc_auc", "brier")


def paired(a: pd.DataFrame, b: pd.DataFrame, metric: str) -> dict | None:
    """Test arm A against arm B over the folds they share."""
    keys = ["repeat", "fold"]
    m = a[keys + [metric, "n_train", "n_test"]].merge(
        b[keys + [metric]], on=keys, suffixes=("_a", "_b"))
    if len(m) < 2:
        return None

    d = (m[f"{metric}_a"] - m[f"{metric}_b"]).to_numpy()
    t, p_t = stats.nadeau_bengio(d, int(m["n_train"].mean()), int(m["n_test"].mean()))
    w, p_w = stats.wilcoxon(d)
    ci = stats.summarize(d)
    return {"metric": metric, "k": len(m), "mean_a": float(m[f"{metric}_a"].mean()),
            "mean_b": float(m[f"{metric}_b"].mean()), "delta": ci["mean"],
            "delta_ci_lo": ci["ci_lo"], "delta_ci_hi": ci["ci_hi"],
            "t_nb": t, "p_nb": p_t, "w": w, "p_wilcoxon": p_w}


def compare_arms(df: pd.DataFrame, metrics) -> list[dict]:
    """Every arm pair, at every caption length.

    Setups are matched, not crossed -- w07 against w07 -- except where one side
    has no length at all. Tags are what the uploader wrote, so the tag arm has
    the single setup `tags`, and each caption length is compared against it.
    Grouping by setup instead would silently drop every tags comparison, since
    `w07` and `tags` never land in the same group.
    """
    out = []
    for (pair, level), g in df.groupby(["pair", "level"], sort=False):
        by_arm = {a: sub for a, sub in g.groupby("arm", sort=False)}
        for family, a, b in ARM_FAMILIES:
            if a not in by_arm or b not in by_arm:
                continue
            setups_b = sorted(by_arm[b]["setup"].unique())
            for setup in sorted(by_arm[a]["setup"].unique()):
                if setup in setups_b:
                    partner = setup
                elif len(setups_b) == 1:
                    partner = setups_b[0]
                else:
                    continue
                for metric in metrics:
                    res = paired(by_arm[a][by_arm[a].setup == setup],
                                 by_arm[b][by_arm[b].setup == partner], metric)
                    if res:
                        out.append({"family": family, "pair": pair,
                                    "setup": setup, "level": level,
                                    "arm_a": a, "arm_b": b, **res})
    return out


def compare_lengths(df: pd.DataFrame, metrics) -> list[dict]:
    """Caption length against length, inside one arm -- w03 is the reference."""
    out = []
    caption_arms = [a for a in df["arm"].unique() if a != "tags"]
    for (pair, arm, level), g in df[df["arm"].isin(caption_arms)].groupby(
            ["pair", "arm", "level"], sort=False):
        by_setup = {s: sub for s, sub in g.groupby("setup", sort=False)}
        if "w03" not in by_setup:
            continue
        for setup in sorted(by_setup):
            if setup == "w03":
                continue
            for metric in metrics:
                res = paired(by_setup[setup], by_setup["w03"], metric)
                if res:
                    out.append({"family": "length", "pair": pair, "setup": setup,
                                "level": level, "arm_a": f"{arm}:{setup}",
                                "arm_b": f"{arm}:w03", **res})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", action="append")
    ap.add_argument("--level", choices=["caption", "image"], action="append")
    args = ap.parse_args()

    metrics = args.metric or list(DEFAULT_METRICS)
    df = pd.read_csv(METRICS_DIR / "folds.csv")
    if args.level:
        df = df[df["level"].isin(args.level)]

    rows = compare_arms(df, metrics) + compare_lengths(df, metrics)
    out = pd.DataFrame(rows)

    # Bonferroni within family x metric x level -- the family is the set of
    # tests that answer one question, so that is the set to correct over.
    for col, dest in (("p_nb", "p_nb_bonf"), ("p_wilcoxon", "p_wilcoxon_bonf")):
        out[dest] = out.groupby(["family", "metric", "level"])[col].transform(
            lambda s: stats.bonferroni(s.to_numpy()))
    out["significant"] = out["p_nb_bonf"] < 0.05

    cols = ["family", "pair", "setup", "level", "arm_a", "arm_b", "metric", "k",
            "mean_a", "mean_b", "delta", "delta_ci_lo", "delta_ci_hi",
            "t_nb", "p_nb", "p_nb_bonf", "w", "p_wilcoxon", "p_wilcoxon_bonf",
            "significant"]
    out[cols].to_csv(METRICS_DIR / "comparisons.csv", index=False)
    print(f"wrote {METRICS_DIR / 'comparisons.csv'} ({len(out)} tests)")

    show = out[(out.level == "image") & (out.metric == "accuracy")]
    for family in show["family"].unique():
        f = show[show.family == family]
        print(f"\n{family}  (image accuracy)")
        for _, r in f.iterrows():
            flag = "*" if r.significant else " "
            print(f"  {flag} {r.pair:22s} {r.setup:5s} {r.arm_a:24s} - {r.arm_b:24s} "
                  f"{r.delta:+.4f} [{r.delta_ci_lo:+.4f},{r.delta_ci_hi:+.4f}] "
                  f"p={r.p_nb_bonf:.2e}")


if __name__ == "__main__":
    main()

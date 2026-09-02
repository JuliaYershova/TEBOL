#!/usr/bin/env python3
"""Stage 3d -- one worked SMER example per class, in the paper's table form.

Reproduces the layout of the SMER table in the paper: every word of one
caption with its embedding, its logit contribution, and the class
probabilities that follow -- then the same for the averaged caption vector,
which is what the classifier actually sees.

The last row is the point of the table. `SAVe` carries

    z = mean(z_j)          logit = z + b          P = expit(logit)

so the per-word rows are not a separate analysis sitting beside the
prediction: they average, exactly, to it. Reading down the z column and
dividing by n reconstructs the caption logit to floating-point error.

Columns follow the paper:

    z_j              beta . e(word_j), no intercept
    b_0              the fitted intercept, constant down the table
    logit(w_j)       z_j + b_0 -- what the model would output for that word
                     alone, since a one-word caption has itself as its mean
    P(.)             expit of that logit, and its complement

**Example choice is mechanical, not curated.** For each class the caption
taken is the one whose predicted probability is closest to that class's
median among correctly-classified captions of the requested length. Picking
the most confident example would flatter the method; the median is a
defensible typical case and the rule is stated so a reader can rerun it.

Coefficients come from the fold that held the caption out, so the numbers in
the table were produced by a model that never saw this image.

    python scripts/06_worked_examples.py
    python scripts/06_worked_examples.py --arm caption_noclass --setup 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tebol import arms as arms_mod  # noqa: E402

PAIRS = ("acousticguitar_violin", "ambulance_firetruck", "ant_bee",
         "cucumber_zucchini", "hotpot_vase")
OUT_DIR = ROOT / "results" / "metrics" / "stage3"


def embedding_preview(vec: np.ndarray, head: int = 2) -> str:
    """[first, second, ..., last] -- enough to show it is a real vector."""
    parts = [f"{v:.4f}" for v in vec[:head]] + ["..."] + [f"{vec[-1]:.4f}"]
    return "[" + ", ".join(parts) + "]"


def pick(df: pd.DataFrame, cls: str, n_words: int) -> pd.DataFrame | None:
    """The median correctly-classified caption of `cls` at that length."""
    cand = df[(df.cls == cls) & (df.pred_class == cls) & (df.n_words == n_words)]
    if cand.empty:
        cand = df[(df.cls == cls) & (df.pred_class == cls)]
        if cand.empty:
            return None

    per = cand.groupby(["stem", "rep"], sort=False)["sent_prob"].first()
    target = per.median()
    stem, rep = per.sub(target).abs().idxmin()
    return cand[(cand.stem == stem) & (cand.rep == rep)].sort_values("pos")


def table(ex: pd.DataFrame, store, neg: str, pos: str) -> list[str]:
    """One class's block, words then the averaged-vector row."""
    bias = float(ex["bias"].iloc[0])
    rows = [
        f"| Word | Embedding | z_j | b_0 | Logit lambda(w_j) "
        f"| P({neg}) | P({pos}) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in ex.itertuples():
        vec = store.vectors[store.row[r.word]]
        p_pos = float(r.word_prob)
        rows.append(
            f"| {r.word} | {embedding_preview(vec)} | {r.z:.4f} | {bias:.4f} "
            f"| {r.z + bias:.4f} | {1 - p_pos:.4f} | {p_pos:.4f} |")

    z_mean = float(ex["z"].mean())
    logit = float(ex["sent_logit"].iloc[0])
    p_pos = float(ex["sent_prob"].iloc[0])
    mean_vec = np.mean([store.vectors[store.row[w]] for w in ex["word"]], axis=0)
    rows += [
        f"| **SAVe(caption)** | {embedding_preview(mean_vec)} | {z_mean:.4f} "
        f"| {bias:.4f} | {logit:.4f} | {1 - p_pos:.4f} | {p_pos:.4f} |",
    ]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="caption", choices=sorted(arms_mod.ARMS))
    ap.add_argument("--setup", type=int, default=7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    emb = arms_mod.ARMS[args.arm].embedding
    setup = f"w{args.setup:02d}" if args.setup else "tags"

    lines = [
        f"# Worked SMER examples -- arm `{args.arm}`, {args.setup}-word captions",
        "",
        "One caption per class. `z_j = beta . e(word_j)`; `b_0` is the fitted "
        "intercept; `logit lambda(w_j) = z_j + b_0` is what the model outputs "
        "for that word alone. The final row is the averaged caption vector, "
        "`SAVe`, and its `z` is the mean of the column above it -- so "
        "`mean(z_j) + b_0` is the caption logit exactly.",
        "",
        "Captions are chosen mechanically: the median correctly-classified "
        "caption of each class at this length. Coefficients come from the "
        "cross-validation fold that held the caption out.",
        "",
    ]

    for pair in PAIRS:
        path = (ROOT / "artifacts" / "explanations" / emb / pair / args.arm
                / setup / "smer_words.parquet")
        if not path.exists():
            print(f"missing {path}, skipped")
            continue
        df = pd.read_parquet(path)
        data = arms_mod.load_arm(ROOT, pair, args.arm, args.setup)
        neg, pos = sorted(set(df["cls"]))

        for cls in (neg, pos):
            ex = pick(df, cls, args.setup)
            if ex is None:
                continue
            caption = " ".join(ex["word"])
            lines += [
                f"## {pair} -- true class `{cls}`",
                "",
                f"> {caption}",
                "",
                f"`{ex['stem'].iloc[0]}`, rep {int(ex['rep'].iloc[0])}, "
                f"held out in fold {int(ex['fold'].iloc[0])}. "
                f"Predicted **{ex['pred_class'].iloc[0]}** "
                f"at P = {max(float(ex['sent_prob'].iloc[0]), 1 - float(ex['sent_prob'].iloc[0])):.4f}.",
                "",
                *table(ex, data.store, neg, pos),
                "",
            ]

    out = Path(args.out) if args.out else OUT_DIR / f"smer_examples_{args.arm}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

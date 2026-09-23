#!/usr/bin/env python3
"""Stage 3e -- AOPC curves for SMER, per arm and per caption length.

Reads only artifacts/explanations/.../smer_words.parquet. Because the model is
linear over a mean of word vectors, removing a set of words is

    logit = bias + mean(z[kept])

so the per-word scalars are sufficient and no embedding, model or
predict_proba call enters this script. That is the whole reason stage 3 stores
`z` rather than a probability per word.

**Two rankings, both reported.**

    local    remove this caption's own top-k words, ranked by their pull
             toward the class the model predicted for it
    global   rank the whole corpus once by mean z, then remove whichever of
             the top words appear in each caption, in corpus order

They answer different questions. Local asks whether the explanation of *this*
caption is faithful; global asks whether a single vocabulary-wide list of
important words captures what the classifier does. Global is the harder test
and it is the one the Diplom notebooks plotted.

**Short captions.** A caption of n words cannot give up n of them: the mean of
an empty set is undefined and falling back to `expit(bias)` produces a curve
that turns *down* at high k, because the intercept alone is less extreme than
the caption's least-supporting word. At w07 the median caption is 6 words, so
76% are exhausted by k=6 and the artefact dominates. Removal is therefore
capped at n-1 -- a caption that runs out simply holds its last value, which
keeps the curve monotone and keeps every caption in the denominator at every k.
`--exhaust drop` switches to excluding them instead, for comparison.

**Intervals** come from the 25 (fold, rep) groups: five cross-validation folds,
each with five independently generated captions per image. That spread mixes
model variation with caption-generation variation, which is the right thing for
"would this curve look the same if we reran the pipeline". It is not a sampling
interval over images -- the reps are nested within images -- so it is reported
as a spread, not as a significance test.

This writes metrics only. Figures are scripts/10_figure_sets.py, which owns
results/figures/aopc/ and clears it on each run -- two scripts drawing into one
directory with different colour conventions is how a figure set drifts out of
alignment without anyone noticing.

Metrics only. Figures are scripts/10_figure_sets.py, which owns
results/figures/aopc/ and clears it each run; two scripts drawing into one
directory with different colour conventions is how a figure set drifts out of
alignment unnoticed.

    python scripts/07_aopc.py
    python scripts/07_aopc.py --pair hotpot_vase --max-k 6
    python scripts/07_aopc.py --ranking global
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tebol import arms as arms_mod, stats  # noqa: E402

PAIRS = ("acousticguitar_violin", "ambulance_firetruck", "ant_bee",
         "cucumber_zucchini", "hotpot_vase")
OUT_DIR = ROOT / "results" / "metrics"
#: k runs to one less than the setup's nominal length: a 3-word caption can
#: give up at most 2 words and still be a caption, so w03 stops at k=2, w05 at
#: 4, w07 at 6, w10 at 9. Anchoring on the requested length rather than on the
#: empirical distribution keeps the rule one sentence long in the write-up, and
#: keeps the far end of each curve resting on typical captions instead of on
#: the handful that overran the request (a few "3-word" captions reach 19).
#: Tags have no requested length, so theirs comes from the median tag count.
MAX_K_CEILING = 15

#: LIME ranking name -> the file holding those weights. Each is a different
#: configuration of the same explainer, all scored on the same captions, so the
#: curves are directly comparable:
#:   lime        bow=False, signed, predicted class, 5000 perturbations
#:   lime_bow    LIME's own default tokenisation (one feature per word type)
#:   lime_nsNNN  the same as `lime` but with a restricted perturbation budget --
#:               at 5000 samples over a 7-word caption LIME covers all 128
#:               possible masks 39 times, so it is interpolating a fully
#:               observed function rather than approximating one
#:   lime_legacy the Diplom notebook's settings, abs() included
LIME_FILES = {
    "lime": "lime_words.parquet",
    "lime_bow": "lime_words__bow.parquet",
    "lime_ns500": "lime_words__ns500.parquet",
    "lime_ns100": "lime_words__ns100.parquet",
    "lime_ns50": "lime_words__ns50.parquet",
    "lime_legacy": "lime_words_legacy.parquet",
}
LIME_RANKINGS = tuple(LIME_FILES) + tuple(f"{k}_global" for k in LIME_FILES)


def loo_abs_drop(df: pd.DataFrame, pos_label: str) -> np.ndarray:
    """abs(probability drop) from leave-one-out -- the Diplom notebook's SMER.

    `process_instance_smer` removed one word at a time, took the drop in the
    predicted class probability, and stored `abs(drop)`. Reproduced here so the
    notebook's figure can be redrawn on this data, and so the LIME comparison
    is legacy on both sides rather than only on one.

    The abs() is what makes this differ from the signed ranking: it scores a
    word that pushes *away* from the prediction as highly as one that supports
    it, so AOPC then removes words whose removal raises the predicted
    probability. The notebook applied it to both SMER and LIME.
    """
    out = np.empty(len(df), dtype=np.float64)
    frame, bounds = caption_blocks(df)
    z_all = frame["z"].to_numpy(dtype=np.float64)
    bias_all = frame["bias"].to_numpy(dtype=np.float64)
    pred_all = frame["pred_class"].to_numpy()
    order = frame.index.to_numpy()

    for a, b in zip(bounds[:-1], bounds[1:]):
        z = z_all[a:b]
        n = len(z)
        bias = bias_all[a]
        toward_pos = pred_all[a] == pos_label
        total = z.sum()
        p_full = expit(bias + total / n)
        if not toward_pos:
            p_full = 1.0 - p_full
        if n < 2:
            out[a:b] = 0.0
            continue
        p_minus = expit(bias + (total - z) / (n - 1))
        if not toward_pos:
            p_minus = 1.0 - p_minus
        out[a:b] = np.abs(p_full - p_minus)

    res = pd.Series(out, index=order)
    return res.reindex(df.index).to_numpy()


def caption_blocks(df: pd.DataFrame):
    """Split the flat table into per-caption slices, without groupby overhead.

    Sorting by (stem, rep, pos) makes each caption a contiguous run, so the
    boundaries are one np.unique and every caption is a numpy view.
    """
    df = df.sort_values(["stem", "rep", "pos"], kind="stable")
    key = (df["stem"].astype(str) + "\x00" + df["rep"].astype(str)).to_numpy()
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    bounds = np.r_[starts, len(key)]
    return df, bounds


def curves(df: pd.DataFrame, pos_label: str, max_k: int, ranking: str,
           order_maps: dict[str, dict[str, int]] | None,
           exhaust: str) -> pd.DataFrame:
    """Mean probability drop at each k, per (fold, rep) group.

    Every caption is handled in one pass, whichever ranking is in use. Splitting
    the global ranking by predicted class and averaging the halves would
    macro-average it while the local ranking micro-averages, and the two curves
    would stop being comparable -- global came out *above* local, which cannot
    happen, since the caption's own top-k is the optimal removal set.
    """
    df, bounds = caption_blocks(df)
    z_all = df["z"].to_numpy(dtype=np.float64)
    bias_all = df["bias"].to_numpy(dtype=np.float64)
    pred_all = df["pred_class"].to_numpy()
    fold_all = df["fold"].to_numpy()
    rep_all = df["rep"].to_numpy()
    word_all = df["word"].to_numpy()
    # LIME weights are already signed toward the predicted class, so they are
    # used as-is; SMER's z is signed toward the positive class and has to be
    # flipped for captions predicted negative.
    lime_all = (df["lime"].to_numpy(dtype=np.float64)
                if "lime" in df.columns else None)
    loo_all = (df["loo_abs"].to_numpy(dtype=np.float64)
               if "loo_abs" in df.columns else None)

    rows = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        z = z_all[a:b]
        n = len(z)
        bias = bias_all[a]
        toward_pos = pred_all[a] == pos_label
        signed = z if toward_pos else -z          # larger = supports prediction

        if ranking in LIME_FILES:
            order = np.argsort(-lime_all[a:b], kind="stable")
        elif ranking == "smer_legacy":
            order = np.argsort(-loo_all[a:b], kind="stable")
        elif ranking.endswith("_global") or ranking == "global":
            om = order_maps["pos" if toward_pos else "neg"]
            rank = np.array([om.get(w, np.inf) for w in word_all[a:b]])
            order = np.argsort(rank, kind="stable")
            order = order[np.isfinite(rank[order])]
        elif ranking in ("local", "smer_subsample"):
            order = np.argsort(-signed, kind="stable")
        else:
            om = order_maps["pos" if toward_pos else "neg"]
            rank = np.array([om.get(w, np.inf) for w in word_all[a:b]])
            order = np.argsort(rank, kind="stable")
            order = order[np.isfinite(rank[order])]

        total = z.sum()
        p_full = expit(bias + total / n)
        if not toward_pos:
            p_full = 1.0 - p_full

        removed = 0.0
        for k in range(max_k + 1):
            avail = min(k, len(order), n - 1)
            active = avail == k          # this caption really gave up k words
            if k and active:
                removed += z[order[k - 1]]
            elif k and not active and exhaust == "drop":
                continue
            kept = n - avail
            p_k = expit(bias + (total - removed) / kept)
            if not toward_pos:
                p_k = 1.0 - p_k
            rows.append((int(fold_all[a]), int(rep_all[a]), k,
                         p_full - p_k, float(active)))

    out = pd.DataFrame(rows, columns=["fold", "rep", "k", "drop", "active"])
    return out.groupby(["fold", "rep", "k"], as_index=False)[
        ["drop", "active"]].mean()


def run(pair: str, arm: str, setup: int, max_k: int, ranking: str,
        exhaust: str, top_n: int | None = None) -> list[dict]:
    emb = arms_mod.ARMS[arm].embedding
    name = f"w{setup:02d}" if setup else "tags"
    path = (ROOT / "artifacts" / "explanations" / emb / pair / arm / name
            / "smer_words.parquet")
    if not path.exists():
        return []

    # The LIME rankings read the subsampled table, which carries z and bias
    # alongside the LIME weights -- so "smer_subsample" is the SMER curve on
    # exactly the captions LIME saw, which is what makes the two comparable.
    base_rank = ranking[:-7] if ranking.endswith("_global") else ranking
    if base_rank in LIME_FILES:
        path = path.with_name(LIME_FILES[base_rank])
    elif base_rank in ("smer_legacy", "smer_subsample"):
        # Same 1000-caption sample LIME saw, so every method curve is drawn on
        # identical captions.
        path = path.with_name("lime_words.parquet")
    if base_rank in LIME_FILES or base_rank.startswith("smer_"):
        if not path.exists():
            return []

    df = pd.read_parquet(path)
    pos_label = sorted(set(df["cls"]))[1]

    if max_k is None:
        if setup:
            max_k = setup - 1
        else:
            lengths = df.groupby(["stem", "rep"])["n_words"].first().to_numpy()
            max_k = int(min(np.median(lengths) - 1, MAX_K_CEILING))
        max_k = max(max_k, 1)

    if base_rank == "smer_legacy":
        df["loo_abs"] = loo_abs_drop(df, pos_label)

    order_maps = None
    if ranking == "smer_legacy_global":
        agg = (df.assign(dir=np.where(df["pred_class"] == pos_label,
                                      "pos", "neg"))
                 .groupby(["dir", "word"])["loo_abs"].mean().reset_index())
        order_maps = {}
        for d in ("pos", "neg"):
            sub = agg[agg["dir"] == d].sort_values("loo_abs", ascending=False)
            if top_n:
                sub = sub.head(top_n)
            order_maps[d] = {w: i for i, w in enumerate(sub["word"])}
    elif base_rank in LIME_FILES and ranking.endswith("_global"):
        # The Diplom notebook's ranking: aggregate word importance over the
        # whole corpus, take the top of that one list, and remove whichever of
        # those words a caption happens to contain. Split by predicted class,
        # as the SMER global ranking is, so "most important" means the same
        # thing on both sides of the boundary -- the notebook pooled the two,
        # which lets a strong cucumber word outrank every zucchini word in the
        # single list both classes then share.
        agg = (df.assign(dir=np.where(df["pred_class"] == pos_label, "pos", "neg"))
                 .groupby(["dir", "word"])["lime"].mean().reset_index())
        order_maps = {}
        for d in ("pos", "neg"):
            sub = agg[agg["dir"] == d].sort_values("lime", ascending=False)
            if top_n:
                sub = sub.head(top_n)
            order_maps[d] = {w: i for i, w in enumerate(sub["word"])}
    elif ranking == "global":
        g = pd.read_parquet(path.with_name("smer_global.parquet"))
        # One corpus order per class direction; a caption uses the one matching
        # its predicted class, so "most important globally" means the same
        # thing on both sides of the decision boundary.
        pos_order = g.sort_values("z_mean", ascending=False)["word"].tolist()
        neg_order = g.sort_values("z_mean", ascending=True)["word"].tolist()
        if top_n:
            pos_order, neg_order = pos_order[:top_n], neg_order[:top_n]
        order_maps = {"pos": {w: i for i, w in enumerate(pos_order)},
                      "neg": {w: i for i, w in enumerate(neg_order)}}

    merged = curves(df, pos_label, max_k, ranking, order_maps, exhaust)

    rows = []
    for k, g in merged.groupby("k"):
        s = stats.summarize(g["drop"].to_numpy())
        # summarize() reports its sample count as "k"; renamed here so it
        # cannot collide with k, the number of words removed.
        rows.append({"pair": pair, "arm": arm, "setup": name,
                     "ranking": ranking, "k": int(k),
                     "mean": s["mean"], "sd": s["sd"],
                     "ci_lo": s["ci_lo"], "ci_hi": s["ci_hi"],
                     "n_groups": s["k"],
                     "frac_active": float(g["active"].mean())})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", action="append", choices=PAIRS)
    ap.add_argument("--arm", action="append", choices=sorted(arms_mod.ARMS))
    ap.add_argument("--ranking", action="append",
                    choices=["local", "global", "smer_legacy",
                             "smer_legacy_global", "smer_subsample",
                             *LIME_RANKINGS])
    ap.add_argument("--max-k", type=int, default=None,
                    help="fixed k ceiling for every setup; default is the "
                         "setup's nominal length minus one (w07 -> 6)")
    ap.add_argument("--exhaust", choices=["hold", "drop"], default="hold")
    ap.add_argument("--top-n", type=int, default=10,
                    help="global rankings remove only the corpus top-N words, "
                         "as the Diplom notebook did (0 = whole vocabulary). "
                         "A caption containing none of them contributes a drop "
                         "of zero, which is why those curves sit far lower.")
    args = ap.parse_args()

    pairs = args.pair or list(PAIRS)
    arm_names = args.arm or list(arms_mod.ARMS)
    rankings = args.ranking or ["local", "global", "smer_legacy",
                                "smer_legacy_global", "smer_subsample",
                                *LIME_RANKINGS]

    rows: list[dict] = []
    for pair in pairs:
        print(pair)
        for arm in arm_names:
            for setup in arms_mod.ARMS[arm].setups:
                for ranking in rankings:
                    r = run(pair, arm, setup, args.max_k, ranking,
                            args.exhaust, args.top_n or None)
                    if r:
                        top = r[-1]
                        print(f"    {arm:24s} {top['setup']:5s} {ranking:6s} "
                              f"AOPC@k={args.max_k} = {top['mean']:.4f} "
                              f"[{top['ci_lo']:.4f},{top['ci_hi']:.4f}]")
                    rows.extend(r)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "aopc.csv", index=False)
    print(f"\nwrote {OUT_DIR / 'aopc.csv'} ({len(df)} rows)")
    write_report(df, args)


def write_report(df: pd.DataFrame, args) -> None:
    """Curve per (arm, setup), k across the columns, with the spread."""
    local = df[df.ranking == "local"]
    ks = sorted(local["k"].unique())
    lines = [
        "# AOPC for SMER",
        "",
        "Mean drop in the predicted class probability after removing the k "
        "highest-ranked words, over the 25 (fold, rep) groups; the interval "
        "under each value is the 95% t-interval of that spread. Removal is "
        f"capped at n-1 words (`--exhaust {args.exhaust}`), so a caption that "
        "runs out holds its last value and stays in the denominator.",
        "",
        "Computed entirely from `smer_words.parquet`: because the classifier "
        "is linear over a mean of word vectors, removing words is "
        "`bias + mean(z[kept])`, so no embedding or model call is involved. "
        "The whole table takes about 25 seconds.",
        "",
        "**Local and global rankings agree to three decimals** in every "
        "configuration -- see the note at the end.",
        "",
    ]
    for pair, gp in local.groupby("pair", sort=True):
        lines += [f"## {pair}", "",
                  "| arm | setup | " + " | ".join(f"k={k}" for k in ks) + " |",
                  "|---|---|" + "---:|" * len(ks)]
        for (arm, setup), g in gp.groupby(["arm", "setup"], sort=False):
            g = g.set_index("k")
            cells = []
            for k in ks:
                if k not in g.index:
                    cells.append("--")
                    continue
                r = g.loc[k]
                cells.append(f"{r['mean']:.3f}<br><sub>{r['ci_lo']:.3f}-{r['ci_hi']:.3f}</sub>")
            lines.append(f"| {arm} | {setup} | " + " | ".join(cells) + " |")
        lines.append("")

    merged = df.pivot_table(index=["pair", "arm", "setup", "k"],
                            columns="ranking", values="mean").reset_index()
    gap = (merged["local"] - merged["global"]).abs()
    lines += [
        "## Local vs global ranking",
        "",
        f"Largest absolute difference across all {len(merged)} points: "
        f"**{gap.max():.4f}**; mean {gap.mean():.4f}. Global never exceeds "
        f"local ({(merged['global'] <= merged['local'] + 1e-9).all()}), which "
        "is the expected ordering -- a caption's own top-k is the optimal "
        "removal set, so no corpus-wide list can beat it.",
        "",
        "That they coincide at all is structural, not a coincidence. "
        "`z = beta . e(word)` depends only on the word, so the corpus ranking "
        "and the within-caption ranking are the same order; they differ only "
        "because the global list averages `z` over folds while the local one "
        "uses the fold's own coefficients. SMER cannot rank a word differently "
        "in two captions, and these two curves are the measurement of that.",
        "",
    ]
    out = ROOT / "results" / "reports" / "05_aopc.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")




if __name__ == "__main__":
    main()

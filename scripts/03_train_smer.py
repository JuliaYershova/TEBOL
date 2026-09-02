#!/usr/bin/env python3
"""Stage 3 -- fit the classifier, score it, and decompose it into word logits.

One run covers the whole comparison: five pairs x five arms x the caption
lengths each arm has, which is 85 configurations, each fitted 25 times under
repeated stratified group CV.

    python scripts/03_train_smer.py                      # everything
    python scripts/03_train_smer.py --pair hotpot_vase   # one pair
    python scripts/03_train_smer.py --arm tags --arm caption_tagsub
    python scripts/03_train_smer.py --dry-run            # sizes, no fitting

What it writes, and why each file exists:

    results/folds/<pair>.csv
        image -> fold, per repeat. Written once and reused by every arm, which
        is what makes the arm differences paired and therefore testable.

    results/metrics/stage3/folds.csv
        one row per (pair, arm, setup, repeat, fold, level, metric). Long, so
        every later question is a groupby rather than a bespoke script.

    results/metrics/stage3/summary.csv
        the same aggregated to mean / sd / t-interval over the 25 folds, plus
        a bootstrap interval over images at the image level.

    artifacts/models/<emb>/<pair>/<arm>/w<NN>/coefs.npz
        beta and intercept per fold. SMER needs beta; keeping all 25 also gives
        the across-fold spread of any word's score for free.

    artifacts/models/.../oof.parquet
        out-of-fold probability for every caption. Every row is scored by a
        model that never saw its image, so downstream analysis is never fitted
        and explained on the same data -- the flaw in the Diplom notebooks,
        where SMER ran over the full frame including the 2000 training rows.

    artifacts/explanations/<emb>/<pair>/<arm>/w<NN>/smer_words.parquet
        per-word z, probability and additive share. This is the AOPC input:
        removing words from a caption is `bias + mean(z[kept])`, so no
        embedding, model or vector is needed again.

    artifacts/explanations/.../smer_global.parquet
        the corpus word ranking with its across-fold standard deviation.

SMER is written for repeat 0 only. A caption gets one out-of-fold beta per
repeat and five near-identical copies of the table would quintuple it for
nothing; the fold-to-fold variation that matters is already summarised in
smer_global's z_sd, computed across all 25 fits.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tebol import arms as arms_mod  # noqa: E402
from tebol import evaluate, folds as folds_mod, smer, stats  # noqa: E402
from tebol.vectors_io import load_index  # noqa: E402

PAIRS = ("acousticguitar_violin", "ambulance_firetruck", "ant_bee",
         "cucumber_zucchini", "hotpot_vase")

METRICS_DIR = ROOT / "results" / "metrics" / "stage3"
FOLDS_DIR = ROOT / "results" / "folds"

#: max_iter well above the default 100: 2560 dimensions do not converge in 100
#: and sklearn only warns. class_weight because the pairs are unbalanced, most
#: sharply hotpot/vase at 203/270 once tags are required. No scaler -- the word
#: vectors are unit length, and standardising would break the exact additive
#: decomposition that SMER rests on.
LR_KW = dict(max_iter=2000, class_weight="balanced")

SETUP_LABEL = {0: "tags"}


def setup_name(setup: int) -> str:
    return SETUP_LABEL.get(setup, f"w{setup:02d}")


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def ensure_folds(pair: str, rebuild: bool = False) -> pd.DataFrame:
    """Image-level folds for one pair, built once and cached on disk.

    Built from the *caption* index, not the tag index, so the assignment covers
    every image; the tag arms filter this table rather than making their own,
    which keeps them paired with the caption arms image for image.
    """
    path = FOLDS_DIR / f"{pair}.csv"
    if path.exists() and not rebuild:
        return folds_mod.load_folds(path)

    emb = ROOT / "artifacts" / "embeddings" / arms_mod.EMB_FULL
    ix = load_index(emb / "index" / f"{pair}.npz")
    seen: dict[str, str] = {}
    for s, c in zip(ix.stem, ix.cls):
        seen.setdefault(str(s), str(c))
    stems = np.array(sorted(seen))
    df = folds_mod.build_folds(stems, np.array([seen[s] for s in stems]))
    folds_mod.save_folds(path, df)
    return df


def run_config(pair: str, arm: str, setup: int, all_folds: pd.DataFrame,
               dry_run: bool = False) -> dict | None:
    """Fit, score and explain one (pair, arm, setup). Returns a run summary."""
    data = arms_mod.load_arm(ROOT, pair, arm, setup)
    keep = arms_mod.nonempty(data)
    n_dropped = int((~keep).sum())
    idx = np.flatnonzero(keep)
    if idx.size == 0:
        print(f"    {pair}/{arm}/{setup_name(setup)}: no usable text, skipped")
        return None

    stem = data.stem[idx]
    cls = data.cls[idx]
    classes = sorted(set(cls))
    if len(classes) != 2:
        print(f"    {pair}/{arm}/{setup_name(setup)}: {len(classes)} classes, skipped")
        return None
    neg, pos = classes

    label = f"{pair}/{arm}/{setup_name(setup)}"
    if dry_run:
        print(f"    {label}: {idx.size} texts, {len(set(stem))} images, "
              f"{data.lengths[idx].mean():.1f} words/text, dropped {n_dropped}")
        return None

    X = arms_mod.caption_means(data)[idx]
    t0 = time.time()

    n_repeats = int(all_folds["repeat"].max()) + 1
    fold_rows: list[dict] = []
    oof: list[pd.DataFrame] = []
    betas: list[np.ndarray] = []
    biases: list[float] = []
    keys: list[tuple[int, int]] = []
    smer_parts: list[pd.DataFrame] = []
    global_parts: list[pd.DataFrame] = []
    n_iter_max = 0

    for repeat in range(n_repeats):
        assign = folds_mod.fold_of(all_folds, repeat)
        text_fold = np.array([assign.get(s, -1) for s in stem])
        if (text_fold < 0).any():
            missing = sorted({s for s, f in zip(stem, text_fold) if f < 0})[:3]
            raise KeyError(f"{label}: stems absent from folds, e.g. {missing}")

        for fold in sorted(set(text_fold.tolist())):
            te = np.flatnonzero(text_fold == fold)
            tr = np.flatnonzero(text_fold != fold)
            if len(set(cls[tr])) != 2 or te.size == 0:
                continue

            clf = LogisticRegression(**LR_KW).fit(X[tr], cls[tr])
            if list(clf.classes_) != [neg, pos]:
                raise AssertionError(f"{label}: unexpected class order {clf.classes_}")
            p = clf.predict_proba(X[te])[:, 1]
            n_iter_max = max(n_iter_max, int(np.max(clf.n_iter_)))

            beta = clf.coef_.ravel().astype(np.float32)
            bias = float(clf.intercept_[0])
            betas.append(beta)
            biases.append(bias)
            keys.append((repeat, fold))

            for level, (s_, y_, p_) in (
                ("caption", (stem[te], cls[te], p)),
                ("image", evaluate.by_image(stem[te], cls[te], p)),
            ):
                row = evaluate.score(y_, p_, neg, pos)
                row.update(pair=pair, arm=arm, setup=setup_name(setup),
                           repeat=repeat, fold=fold, level=level,
                           n_train=len(tr), n_test=len(te))
                fold_rows.append(row)

            oof.append(pd.DataFrame({
                "stem": stem[te], "rep": data.rep[idx][te], "repeat": repeat,
                "fold": fold, "y_true": cls[te], "p_pos": p.astype(np.float32),
            }))

            z_vocab = smer.vocab_logits(data.store, beta)
            part = smer.explain(data, z_vocab, bias, neg, pos,
                                rows=idx[te], fold=fold, repeat=repeat)
            global_parts.append(part[["word", "stem", "z", "word_prob"]])
            if repeat == 0:
                smer_parts.append(part)

    if not fold_rows:
        print(f"    {label}: no usable fold, skipped")
        return None

    # ---- persist ---------------------------------------------------------
    emb_name = arms_mod.ARMS[arm].embedding
    model_dir = ROOT / "artifacts" / "models" / emb_name / pair / arm / setup_name(setup)
    expl_dir = ROOT / "artifacts" / "explanations" / emb_name / pair / arm / setup_name(setup)
    model_dir.mkdir(parents=True, exist_ok=True)
    expl_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        model_dir / "coefs.npz",
        beta=np.stack(betas), intercept=np.asarray(biases, dtype=np.float32),
        repeat=np.asarray([k[0] for k in keys], dtype=np.int16),
        fold=np.asarray([k[1] for k in keys], dtype=np.int16),
        classes=np.asarray([neg, pos], dtype=object),
    )
    pd.concat(oof, ignore_index=True).to_parquet(model_dir / "oof.parquet", index=False)

    smer_df = pd.concat(smer_parts, ignore_index=True)
    smer_df["imp_pred"] = smer.align(smer_df, "pred", pos)
    smer_df["imp_true"] = smer.align(smer_df, "true", pos)
    smer_df.to_parquet(expl_dir / "smer_words.parquet", index=False)

    ranking = smer.global_ranking(pd.concat(global_parts, ignore_index=True))
    ranking.to_parquet(expl_dir / "smer_global.parquet", index=False)

    meta = {
        "pair": pair, "arm": arm, "setup": setup_name(setup),
        "embedding": emb_name, "dim": data.store.dim,
        "n_texts": int(idx.size), "n_images": len(set(stem)),
        "n_dropped_empty": n_dropped,
        "mean_words": float(data.lengths[idx].mean()),
        "classes": [neg, pos], "lr": {k: str(v) for k, v in LR_KW.items()},
        "max_n_iter": n_iter_max, "converged": n_iter_max < LR_KW["max_iter"],
        "n_fits": len(betas), "seed": folds_mod.SEED,
        "git_sha": git_sha(), "seconds": round(time.time() - t0, 1),
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (model_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    acc = np.mean([r["accuracy"] for r in fold_rows if r["level"] == "image"])
    warn = "" if meta["converged"] else "  ! did not converge"
    print(f"    {label}: {idx.size} texts / {meta['n_images']} images, "
          f"image acc {acc:.4f}, {meta['seconds']}s{warn}")

    return {"fold_rows": fold_rows, "meta": meta}


def summarise(folds_df: pd.DataFrame) -> pd.DataFrame:
    """mean / sd / t-interval across the 25 fits, per metric."""
    skip = {"pair", "arm", "setup", "repeat", "fold", "level", "n", "n_pos",
            "n_train", "n_test", "tn", "fp", "fn", "tp"}
    metrics = [c for c in folds_df.columns if c not in skip]
    out = []
    for (pair, arm, setup, level), g in folds_df.groupby(
            ["pair", "arm", "setup", "level"], sort=False):
        for m in metrics:
            if g[m].isna().all():
                continue
            row = stats.summarize(g[m].to_numpy())
            row.update(pair=pair, arm=arm, setup=setup, level=level, metric=m)
            out.append(row)
    cols = ["pair", "arm", "setup", "level", "metric", "mean", "sd",
            "ci_lo", "ci_hi", "k"]
    return pd.DataFrame(out)[cols]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", action="append", choices=PAIRS)
    ap.add_argument("--arm", action="append", choices=sorted(arms_mod.ARMS))
    ap.add_argument("--setup", action="append", type=int)
    ap.add_argument("--rebuild-folds", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="report sizes without fitting anything")
    args = ap.parse_args()

    pairs = args.pair or list(PAIRS)
    arm_names = args.arm or list(arms_mod.ARMS)

    all_rows: list[dict] = []
    for pair in pairs:
        print(f"\n{pair}")
        pair_folds = ensure_folds(pair, args.rebuild_folds)
        for arm in arm_names:
            setups = [s for s in arms_mod.ARMS[arm].setups
                      if args.setup is None or s in args.setup]
            for setup in setups:
                res = run_config(pair, arm, setup, pair_folds, args.dry_run)
                if res:
                    all_rows.extend(res["fold_rows"])

    if args.dry_run or not all_rows:
        return

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    lead = ["pair", "arm", "setup", "level", "repeat", "fold"]
    folds_df = pd.DataFrame(all_rows)
    folds_df = folds_df[lead + [c for c in folds_df.columns if c not in lead]]
    folds_df.to_csv(METRICS_DIR / "folds.csv", index=False)
    summarise(folds_df).to_csv(METRICS_DIR / "summary.csv", index=False)

    print(f"\nwrote {METRICS_DIR / 'folds.csv'} ({len(folds_df)} rows)")
    print(f"wrote {METRICS_DIR / 'summary.csv'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Stage 3i -- how much of the result is noise, by source.

Three things vary between runs of this pipeline, and they are not the same
thing. Reported separately, because a reader who is told "the explanation is
stable" deserves to know stable against what.

    model       refit on a different cross-validation split -> different beta
                -> different z -> possibly a different word ranking. SMER is
                exact *given* a model, not given a dataset, so this is the axis
                on which SMER can move at all.

    caption     stage 1 generated five captions per image at temperature 1.0.
                Different words, same photograph. This is the captioner's
                randomness, and it moves both methods equally.

    explainer   LIME draws random perturbations, so it can return a different
                answer for the same caption and the same model. SMER cannot:
                z = beta . e involves no sampling. Measured by 12_lime_seeds.py,
                which is the only one of the three that needs new computation.

Metrics, per caption:

    top-k Jaccard   mean pairwise overlap of the top-k word sets over the runs
    top-1 changed   share of runs whose single most important word differs
                    from the majority choice
    rank tau        mean pairwise Kendall tau over the full ordering

Reported alongside the accuracy spread over the same fits, so "how much does
the model move" and "how much does the explanation move" can be read together.

    python scripts/11_stochasticity.py
    python scripts/11_stochasticity.py --pair hotpot_vase --topk 5
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tebol import arms as arms_mod  # noqa: E402
from tebol.vectors_io import load_matrix  # noqa: E402

PAIRS = ("acousticguitar_violin", "ambulance_firetruck", "ant_bee",
         "cucumber_zucchini", "hotpot_vase")
OUT_DIR = ROOT / "results" / "metrics"
TOPK = 3
MAX_CAPTIONS = 400


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


def stability(rankings: list[np.ndarray], topk: int) -> dict:
    """Agreement between several orderings of the same caption's words."""
    tops = [set(r[:topk].tolist()) for r in rankings]
    pairs = list(itertools.combinations(range(len(rankings)), 2))
    jac = float(np.mean([jaccard(tops[i], tops[j]) for i, j in pairs]))
    firsts = [int(r[0]) for r in rankings]
    majority = max(set(firsts), key=firsts.count)
    changed = float(np.mean([f != majority for f in firsts]))
    taus = []
    for i, j in pairs:
        t = kendalltau(rankings[i], rankings[j]).statistic
        if np.isfinite(t):
            taus.append(t)
    return {"jaccard": jac, "top1_changed": changed,
            "tau": float(np.mean(taus)) if taus else np.nan}


def model_stability(pair: str, arm: str, setup: int, topk: int,
                    limit: int) -> dict | None:
    """Refit the model on a different split: does the word ranking move?

    Uses the 25 stored coefficient vectors -- five folds x five repeats -- so
    nothing is retrained here. For each caption, z is recomputed under every
    beta and the resulting orderings compared.
    """
    emb = arms_mod.ARMS[arm].embedding
    name = f"w{setup:02d}" if setup else "tags"
    coefs = ROOT / "artifacts" / "models" / emb / pair / arm / name / "coefs.npz"
    words = (ROOT / "artifacts" / "explanations" / emb / pair / arm / name
             / "smer_words.parquet")
    if not coefs.exists() or not words.exists():
        return None

    with np.load(coefs, allow_pickle=True) as z:
        betas = z["beta"]
        pos_label = str(z["classes"][1])

    store = load_matrix(ROOT / "artifacts" / "embeddings" / emb
                        / arms_mod.ARMS[arm].matrix)
    row = {k: i for i, k in enumerate(store.keys)}
    zv = np.stack([store.vectors @ b.astype(store.vectors.dtype)
                   for b in betas]).astype(np.float64)     # [n_fits, n_vocab]

    df = pd.read_parquet(words).sort_values(["stem", "rep", "pos"])
    groups = [g for _, g in df.groupby(["stem", "rep"], sort=True)][:limit]

    out = []
    for g in groups:
        idx = [row[w] for w in g["word"] if w in row]
        if len(idx) < max(3, topk):
            continue
        toward = 1.0 if str(g["pred_class"].iloc[0]) == pos_label else -1.0
        rankings = [np.argsort(-(zv[f][idx] * toward), kind="stable")
                    for f in range(len(betas))]
        out.append(stability(rankings, topk))
    if not out:
        return None
    r = pd.DataFrame(out).mean().to_dict()
    r.update(pair=pair, arm=arm, setup=name, source="model",
             n_runs=len(betas), n_captions=len(out))
    return r


def caption_stability(pair: str, arm: str, setup: int, topk: int,
                      limit: int) -> dict | None:
    """Regenerate the caption: does the word ranking move?

    The five reps are five independent captions of one photograph, so their
    word lists differ. Overlap is therefore measured over words, not positions,
    and a low number here is the captioner talking about the image differently
    rather than the explainer being unstable.
    """
    emb = arms_mod.ARMS[arm].embedding
    name = f"w{setup:02d}" if setup else "tags"
    path = (ROOT / "artifacts" / "explanations" / emb / pair / arm / name
            / "smer_words.parquet")
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    pos_label = sorted(set(df["cls"]))[1]

    out = []
    for stem, g in itertools.islice(df.groupby("stem", sort=True), limit):
        tops = []
        for _, cap in g.groupby("rep", sort=True):
            toward = 1.0 if str(cap["pred_class"].iloc[0]) == pos_label else -1.0
            s = cap.assign(v=cap["z"] * toward).nlargest(topk, "v")
            tops.append(set(s["word"]))
        if len(tops) < 2:
            continue
        pairs = list(itertools.combinations(range(len(tops)), 2))
        out.append({"jaccard": float(np.mean(
            [jaccard(tops[i], tops[j]) for i, j in pairs]))})
    if not out:
        return None
    r = pd.DataFrame(out).mean().to_dict()
    r.update(pair=pair, arm=arm, setup=name, source="caption",
             n_runs=5, n_captions=len(out), top1_changed=np.nan, tau=np.nan)
    return r


def prediction_stability(pair: str, arm: str, setup: int) -> dict | None:
    """Do the five captions of one image agree on the predicted class?"""
    emb = arms_mod.ARMS[arm].embedding
    name = f"w{setup:02d}" if setup else "tags"
    path = ROOT / "artifacts" / "models" / emb / pair / arm / name / "oof.parquet"
    if not path.exists():
        return None
    o = pd.read_parquet(path)
    o = o[o["repeat"] == 0]
    o["pred"] = o["p_pos"] >= 0.5
    g = o.groupby("stem")["pred"]
    unanimous = float((g.nunique() == 1).mean())
    return {"pair": pair, "arm": arm, "setup": name, "source": "prediction",
            "unanimous_images": unanimous, "n_captions": len(o),
            "n_images": int(g.ngroups)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", action="append", choices=PAIRS)
    ap.add_argument("--arm", action="append", default=None)
    ap.add_argument("--setup", type=int, default=7)
    ap.add_argument("--topk", type=int, default=TOPK)
    ap.add_argument("--limit", type=int, default=MAX_CAPTIONS)
    args = ap.parse_args()

    pairs = args.pair or list(PAIRS)
    arms = args.arm or ["caption", "caption_noclass"]

    rows, preds = [], []
    for pair in pairs:
        for arm in arms:
            for fn in (model_stability, caption_stability):
                r = fn(pair, arm, args.setup, args.topk, args.limit)
                if r:
                    rows.append(r)
            p = prediction_stability(pair, arm, args.setup)
            if p:
                preds.append(p)
        print(f"  {pair} done")

    df = pd.DataFrame(rows)
    cols = ["pair", "arm", "setup", "source", "n_runs", "n_captions",
            "jaccard", "top1_changed", "tau"]
    df = df[[c for c in cols if c in df.columns]]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "stochasticity.csv", index=False)
    pd.DataFrame(preds).to_csv(OUT_DIR / "prediction_stability.csv", index=False)

    print(f"\nSMER top-{args.topk} stability (w{args.setup:02d})")
    print(f"  {'pair':<24}{'arm':<18}{'source':<11}{'Jaccard':>9}"
          f"{'top1 chg':>10}{'tau':>8}")
    for r in df.itertuples():
        t1 = "--" if pd.isna(r.top1_changed) else f"{r.top1_changed:.3f}"
        ta = "--" if pd.isna(r.tau) else f"{r.tau:.3f}"
        print(f"  {r.pair:<24}{r.arm:<18}{r.source:<11}{r.jaccard:9.3f}"
              f"{t1:>10}{ta:>8}")
    print(f"\nwrote {OUT_DIR / 'stochasticity.csv'} and prediction_stability.csv")


if __name__ == "__main__":
    main()

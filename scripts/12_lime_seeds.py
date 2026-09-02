#!/usr/bin/env python3
"""Stage 3j -- LIME's stability, on the same axes SMER was measured on.

Two experiments, matching 11_stochasticity.py so the numbers sit in one table:

    model    the ranking recomputed under each of the 25 stored coefficient
             vectors, caption and seed held fixed. SMER's version of this is
             free (z = beta . e for each beta); LIME's needs a fresh run per
             beta, because the function it samples has changed.

    seed     the same caption and the same beta, with five different values of
             LIME's random_state. This is the axis SMER does not have: z
             involves no sampling, so SMER's numbers here are 1.000 Jaccard,
             0.000 top-1 changes and tau 1.000 by construction. They are
             written into the output as a derivation rather than obtained by
             running a deterministic function five times and reporting that it
             returned the same answer.

Both use the identical metrics from 11_stochasticity -- top-k Jaccard, share of
runs whose top-1 word differs from the majority, mean pairwise Kendall tau --
and the same captions, so the model axis is directly comparable between the two
explainers.

    python scripts/12_lime_seeds.py
    python scripts/12_lime_seeds.py --n 100 --jobs 6
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tebol import arms as arms_mod  # noqa: E402
from tebol.lime_runner import explain_one  # noqa: E402
from tebol.vectors_io import load_matrix  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "stoch", ROOT / "scripts" / "11_stochasticity.py")
_stoch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_stoch)
stability = _stoch.stability

PAIRS = ("acousticguitar_violin", "ambulance_firetruck", "ant_bee",
         "cucumber_zucchini", "hotpot_vase")
OUT_DIR = ROOT / "results" / "metrics" / "stage3"

N_CAPTIONS = 200
NUM_SAMPLES = 5000
BASE_SEED = 20260824
SEEDS = [BASE_SEED + i for i in range(5)]


def lime_rank(words, z, bias, toward_pos, seed, num_samples):
    w = explain_one(words, z, bias, toward_pos, num_samples, seed, False, True)
    return np.argsort(-w, kind="stable")


def run_pair(pair: str, arm: str, setup: int, topk: int, n: int,
             num_samples: int, jobs: int) -> list[dict]:
    emb = arms_mod.ARMS[arm].embedding
    name = f"w{setup:02d}" if setup else "tags"
    coefs = ROOT / "artifacts" / "models" / emb / pair / arm / name / "coefs.npz"
    words_path = (ROOT / "artifacts" / "explanations" / emb / pair / arm / name
                  / "lime_words__bow.parquet")
    if not coefs.exists() or not words_path.exists():
        return []

    with np.load(coefs, allow_pickle=True) as zf:
        betas, biases = zf["beta"], zf["intercept"]
        pos_label = str(zf["classes"][1])

    store = load_matrix(ROOT / "artifacts" / "embeddings" / emb
                        / arms_mod.ARMS[arm].matrix)
    row = {k: i for i, k in enumerate(store.keys)}
    zv = np.stack([store.vectors @ b.astype(store.vectors.dtype)
                   for b in betas]).astype(np.float64)

    df = pd.read_parquet(words_path).sort_values(["stem", "rep", "pos"])
    groups = [g for _, g in df.groupby(["stem", "rep"], sort=True)][:n]

    caps = []
    for g in groups:
        ws = [str(w) for w in g["word"]]
        idx = [row[w] for w in ws if w in row]
        if len(idx) < max(3, topk) or len(idx) != len(ws):
            continue
        caps.append((ws, idx,
                     bool(str(g["pred_class"].iloc[0]) == pos_label),
                     float(g["bias"].iloc[0]),
                     np.asarray(g["z"], dtype=np.float64)))

    t0 = time.time()
    # --- model axis: one LIME run per stored beta -------------------------
    tasks = [(c[0], zv[f][c[1]] , float(biases[f]), c[2], BASE_SEED)
             for c in caps for f in range(len(betas))]
    ranks = Parallel(n_jobs=jobs, batch_size=32)(
        delayed(lime_rank)(w, z, b, tp, sd, num_samples)
        for w, z, b, tp, sd in tasks)
    nf = len(betas)
    model_rows = [stability(ranks[i * nf:(i + 1) * nf], topk)
                  for i in range(len(caps))]

    # --- seed axis: one LIME run per random_state, beta fixed -------------
    tasks = [(c[0], c[4], c[3], c[2], sd) for c in caps for sd in SEEDS]
    ranks = Parallel(n_jobs=jobs, batch_size=32)(
        delayed(lime_rank)(w, z, b, tp, sd, num_samples)
        for w, z, b, tp, sd in tasks)
    ns = len(SEEDS)
    seed_rows = [stability(ranks[i * ns:(i + 1) * ns], topk)
                 for i in range(len(caps))]

    out = []
    for src, rws, nruns in (("model", model_rows, nf), ("seed", seed_rows, ns)):
        r = pd.DataFrame(rws).mean().to_dict()
        r.update(pair=pair, arm=arm, setup=name, method="LIME", source=src,
                 n_runs=nruns, n_captions=len(caps))
        out.append(r)
    # SMER on the seed axis is not measured: z = beta . e is deterministic, so
    # five runs return five identical rankings. Recorded as the derivation.
    out.append({"jaccard": 1.0, "top1_changed": 0.0, "tau": 1.0,
                "pair": pair, "arm": arm, "setup": name, "method": "SMER",
                "source": "seed", "n_runs": len(SEEDS), "n_captions": len(caps)})
    print(f"    {pair}/{arm}: {len(caps)} captions, {time.time() - t0:.0f}s")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", action="append", choices=PAIRS)
    ap.add_argument("--arm", action="append",
                    default=None)
    ap.add_argument("--setup", type=int, default=7)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--n", type=int, default=N_CAPTIONS)
    ap.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()

    rows = []
    for pair in args.pair or list(PAIRS):
        for arm in args.arm or ["caption", "caption_noclass"]:
            rows.extend(run_pair(pair, arm, args.setup, args.topk, args.n,
                                 args.num_samples, args.jobs))

    df = pd.DataFrame(rows)
    cols = ["pair", "arm", "setup", "method", "source", "n_runs",
            "n_captions", "jaccard", "top1_changed", "tau"]
    df = df[cols]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "stochasticity_lime.csv", index=False)
    print(f"\nwrote {OUT_DIR / 'stochasticity_lime.csv'} ({len(df)} rows)")

    print(f"\n{'pair':<24}{'arm':<18}{'method':<7}{'source':<8}"
          f"{'Jaccard':>9}{'top1 chg':>10}{'tau':>8}")
    for r in df.itertuples():
        print(f"{r.pair:<24}{r.arm:<18}{r.method:<7}{r.source:<8}"
              f"{r.jaccard:9.3f}{r.top1_changed:10.3f}{r.tau:8.3f}")


if __name__ == "__main__":
    main()

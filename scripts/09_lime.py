#!/usr/bin/env python3
"""Stage 3g -- LIME word attributions, for the comparison against SMER.

LIME is cheap here for a reason worth stating. Its perturbation loop needs the
classifier's output on thousands of word-subsets, and for this model that is

    p = expit(bias + mean(z[kept]))

so every one of those calls is arithmetic on scalars stage 3 already stored.
No embedding, no API, no predict_proba. The cost is LIME's own bookkeeping,
about 77 ms per caption at the default 5000 samples.

**bow=False, deliberately.** With the default `bow=True` LIME treats a word as
one feature wherever it occurs and removes every copy together -- the same
defect the Diplom notebooks had in their SMER loop (`w != word` deleting both
copies of "side"). With bow=False each position is its own feature, matching
how SMER scores them, so the two methods rank the same objects. LIME marks a
removed position with the sentinel `UNKWORDZ` rather than deleting it, so the
mask is recoverable from the perturbed string.

**The predicted class, not the true one.** LIME is asked to explain whichever
class the model actually chose, so a positive weight always means "supports the
prediction" -- the same convention SMER uses, and the one AOPC requires.

**Subsampling.** The full corpus is 30 hours of LIME. A fixed sample of
captions per configuration is drawn instead, stratified by fold and class and
seeded, and the sampled keys are written alongside the weights so the SMER
curve can be recomputed on exactly the same captions. A comparison between two
explainers on different caption sets would not be a comparison.

    python scripts/09_lime.py                       # every configuration
    python scripts/09_lime.py --pair hotpot_vase --n 200
    python scripts/09_lime.py --jobs 4 --num-samples 1000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import expit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tebol import arms as arms_mod  # noqa: E402
from tebol.lime_runner import explain_one  # noqa: E402

PAIRS = ("acousticguitar_violin", "ambulance_firetruck", "ant_bee",
         "cucumber_zucchini", "hotpot_vase")

N_CAPTIONS = 1000
NUM_SAMPLES = 5000
SEED = 20260824


def sample_captions(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """A seeded, fold- and class-stratified subset of captions."""
    keys = df.groupby(["stem", "rep"], sort=True).first().reset_index()
    if len(keys) <= n:
        return keys[["stem", "rep"]]
    frac = n / len(keys)
    picked = (keys.groupby(["fold", "cls"], group_keys=False)
              .apply(lambda g: g.sample(max(1, int(round(len(g) * frac))),
                                        random_state=seed),
                     include_groups=False)
              .reset_index())
    return picked[["stem", "rep"]]


def variant_suffix(legacy: bool, bow: bool, num_samples: int) -> str:
    """File suffix identifying this LIME configuration."""
    if legacy:
        return "_legacy"
    bits = []
    if bow:
        bits.append("bow")
    if num_samples != NUM_SAMPLES:
        bits.append(f"ns{num_samples}")
    return ("__" + "_".join(bits)) if bits else ""


def run_config(pair: str, arm: str, setup: int, n: int, num_samples: int,
               jobs: int, overwrite: bool, legacy: bool = False,
               bow: bool = False) -> str:
    emb = arms_mod.ARMS[arm].embedding
    name = f"w{setup:02d}" if setup else "tags"
    base = ROOT / "artifacts" / "explanations" / emb / pair / arm / name
    src = base / "smer_words.parquet"
    dst = base / f"lime_words{variant_suffix(legacy, bow, num_samples)}.parquet"
    if not src.exists():
        return f"{pair}/{arm}/{name}: no smer_words, skipped"
    if dst.exists() and not overwrite:
        return f"{pair}/{arm}/{name}: exists, skipped (--overwrite to redo)"

    df = pd.read_parquet(src).sort_values(["stem", "rep", "pos"])
    pos_label = sorted(set(df["cls"]))[1]
    picked = sample_captions(df, n, SEED)
    df = df.merge(picked, on=["stem", "rep"], how="inner")

    # Plain str / float64 across the process boundary: pandas and numpy scalar
    # types do not all survive loky's pickling, and the failure surfaces as an
    # opaque BrokenProcessPool rather than as a type error.
    tasks = []
    for _, g in df.groupby(["stem", "rep"], sort=True):
        tasks.append(([str(w) for w in g["word"]],
                      np.asarray(g["z"], dtype=np.float64),
                      float(g["bias"].iloc[0]),
                      bool(str(g["pred_class"].iloc[0]) == pos_label)))

    t0 = time.time()
    weights = Parallel(n_jobs=jobs, batch_size=16)(
        delayed(explain_one)(w, z, b, tp, num_samples, SEED, legacy, bow)
        for w, z, b, tp in tasks)
    groups = tasks

    out = df.copy()
    out["lime"] = np.concatenate(weights).astype(np.float32)
    out.to_parquet(dst, index=False)
    return (f"{pair}/{arm}/{name}: {len(groups)} captions, "
            f"{time.time() - t0:.0f}s -> {dst.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", action="append", choices=PAIRS)
    ap.add_argument("--arm", action="append", choices=sorted(arms_mod.ARMS))
    ap.add_argument("--n", type=int, default=N_CAPTIONS,
                    help="captions sampled per configuration")
    ap.add_argument("--num-samples", type=int, default=NUM_SAMPLES,
                    help="LIME perturbations per caption")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--bow", action="store_true",
                    help="LIME's default bag-of-words tokenisation: one "
                         "feature per distinct word, all copies removed at once")
    ap.add_argument("--legacy", action="store_true",
                    help="reproduce the Diplom notebook's LIME settings: "
                         "bow=True, abs(), labels=(1,), num_features=15")
    args = ap.parse_args()

    for pair in args.pair or list(PAIRS):
        print(pair)
        for arm in args.arm or list(arms_mod.ARMS):
            for setup in arms_mod.ARMS[arm].setups:
                print("   ", run_config(pair, arm, setup, args.n,
                                        args.num_samples, args.jobs,
                                        args.overwrite, args.legacy, args.bow))


if __name__ == "__main__":
    main()

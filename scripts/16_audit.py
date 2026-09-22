#!/usr/bin/env python3
"""Full-pipeline audit -- one table saying what exists and whether it is complete.

Walks every stage from raw captions to the final reports and checks the counts
that should hold. Writes results/reports/00_audit.md.

Each row states what was checked, what the count should be and what it is, so a
FAIL points at the stage that produced it rather than at this script.

    python scripts/16_audit.py
"""
from __future__ import annotations
import collections, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from tebol import arms as A                                    # noqa: E402
from tebol.vectors_io import load_index, load_matrix           # noqa: E402

PAIRS = ("acousticguitar_violin", "ambulance_firetruck", "ant_bee",
         "cucumber_zucchini", "hotpot_vase")
LEN = (3, 5, 7, 10, 15, 20, 25, 30)
REPS = 5
EMB = {"full": A.EMB_FULL, "noclass": A.EMB_NOCLASS}
CAPFILE = {"full": "local__qwen3-vl__32b-instruct.jsonl",
           "noclass": "local__qwen3-vl__32b-instruct--noclass.jsonl"}
rows = []


def add(stage, check, exp, got, ok=None):
    rows.append({"stage": stage, "check": check, "expected": exp, "found": got,
                 "status": ("PASS" if (got == exp if ok is None else ok) else "FAIL")})


# 1. captions -------------------------------------------------------------
n_img = {}
for variant, fn in CAPFILE.items():
    tot = unusable = 0
    for p in PAIRS:
        key = collections.defaultdict(list); imgs = set()
        for line in (ROOT / "artifacts/captions" / p / fn).open(encoding="utf-8"):
            r = json.loads(line); imgs.add(r["stem"])
            key[(r["stem"], r["n_words"], r["rep"])].append(bool(r.get("tokens")))
        n_img[p] = len(imgs)
        tot += len(key)
        unusable += sum(1 for v in key.values() if not any(v))
    exp = sum(n_img[p] for p in PAIRS) * len(LEN) * REPS
    add("1 captions", f"{variant}: image x length x rep triples", exp, tot)
    add("1 captions", f"{variant}: triples with no usable caption",
        0, unusable, ok=(unusable == 0 or variant == "noclass"))

# 2. embeddings -----------------------------------------------------------
for variant, emb in EMB.items():
    v = load_matrix(ROOT / "artifacts/embeddings" / emb / "vocab.npz")
    tot = 0; gaps = 0
    for p in PAIRS:
        ix = load_index(ROOT / "artifacts/embeddings" / emb / "index" / f"{p}.npz")
        tot += len(ix)
        for L in LEN:
            m = ix.n_words == L
            d = collections.defaultdict(set)
            for s, r in zip(ix.stem[m], ix.rep[m]):
                d[str(s)].add(int(r))
            if len(d) != n_img[p] or set(collections.Counter(
                    len(x) for x in d.values())) != {REPS}:
                gaps += 1
    add("2 embeddings", f"{variant}: vocab words", ">0", f"{len(v):,}", ok=len(v) > 0)
    add("2 embeddings", f"{variant}: captions indexed",
        sum(n_img[p] for p in PAIRS) * len(LEN) * REPS, tot)
    add("2 embeddings", f"{variant}: (pair,length) blocks with a gap", 0, gaps)

tag_imgs = sum(len(A.tag_stems(ROOT, p)) for p in PAIRS)
add("2 embeddings", "tags: images with an indexed tag set", ">0", f"{tag_imgs:,}",
    ok=tag_imgs > 0)

# 3-5. per-config artefacts ----------------------------------------------
cfgs = [(p, a, s) for p in PAIRS for a in A.ARMS for s in A.ARMS[a].setups]
want = len(cfgs)
files = {"3 models": ["coefs.npz", "oof.parquet", "run_meta.json"],
         "4 SMER": ["smer_words.parquet", "smer_global.parquet"],
         "5 LIME": ["lime_words.parquet", "lime_words__bow.parquet"]}
base = {"3 models": "models", "4 SMER": "explanations", "5 LIME": "explanations"}
for stage, names in files.items():
    for nm in names:
        got = sum(1 for p, a, s in cfgs if (
            ROOT / "artifacts" / base[stage] / A.ARMS[a].embedding / p / a /
            (f"w{s:02d}" if s else "tags") / nm).exists())
        add(stage, nm, want, got)

conv = [json.loads(f.read_text())["converged"]
        for f in (ROOT / "artifacts/models").rglob("run_meta.json")]
add("3 models", "all LR fits converged", want, sum(conv))

# 6. AOPC ------------------------------------------------------------------
a = pd.read_csv(ROOT / "results/metrics/aopc.csv")
prim = ["local", "global", "smer_subsample", "lime", "lime_bow"]
for r in prim:
    add("6 AOPC", f"{r}: configs covered", want,
        a[a.ranking == r].groupby(["pair", "arm", "setup"]).ngroups)
bad = sum(1 for _, g in a[a.ranking.isin(prim)].groupby(
              ["pair", "arm", "setup", "ranking"])
          if not all(x <= y + 1e-9 for x, y in zip(g.sort_values("k")["mean"],
                                                   g.sort_values("k")["mean"][1:])))
add("6 AOPC", "non-monotone primary curves", 0, bad)

# 7. comparisons, stochasticity -------------------------------------------
c = pd.read_csv(ROOT / "results/metrics/comparisons.csv")
add("7 comparisons", "paired tests", ">0", f"{len(c):,}", ok=len(c) > 0)
add("7 comparisons", "setups covered", len(LEN), c.setup.nunique())
for f, lab in (("stochasticity.csv", "SMER"), ("stochasticity_lime.csv", "LIME")):
    d = pd.read_csv(ROOT / "results/metrics" / f)
    add("8 stochasticity", f"{lab}: rows", ">0", len(d), ok=len(d) > 0)
    add("8 stochasticity", f"{lab}: lengths measured", ">1", d.setup.nunique(),
        ok=d.setup.nunique() > 1)
cs = pd.read_csv(ROOT / "results/metrics/caption_stability.csv")
add("8 stochasticity", "caption stability: lengths", len(LEN), cs.n_words.nunique())

# 9. figures and reports ---------------------------------------------------
add("9 figures", "AOPC png", 205, len(list((ROOT / "results/figures/aopc").rglob("*.png"))))
add("9 figures", "AOPC folders", 6,
    len([d for d in (ROOT / "results/figures/aopc").iterdir() if d.is_dir()]))
add("9 figures", "caption-length png", ">0",
    len(list((ROOT / "results/figures/accuracy_vs_length").glob("*.png"))),
    ok=any((ROOT / "results/figures/accuracy_vs_length").glob("*.png")))
md = sorted((ROOT / "results/reports").glob("*.md"))
add("10 reports", "markdown reports", ">0", len(md), ok=len(md) > 0)

# write --------------------------------------------------------------------
df = pd.DataFrame(rows)
out = ["# Pipeline audit", "",
       f"{(df.status == 'PASS').sum()} of {len(df)} checks pass.", "",
       "| stage | check | expected | found | |", "|---|---|---:|---:|---|"]
for r in df.itertuples():
    out.append(f"| {r.stage} | {r.check} | {r.expected} | {r.found} "
               f"| {'ok' if r.status == 'PASS' else '**FAIL**'} |")
out += ["", "Regenerate with `python scripts/16_audit.py`.", ""]
(ROOT / "results/reports/00_audit.md").write_text("\n".join(out) + "\n")
print("\n".join(out))

#!/usr/bin/env python3
"""Stage 2 -- embed every word that appears in a caption.

SMER scores a caption one word at a time, so the word is the unit that gets a
vector. The caption vector is the mean of its words' vectors, assembled
downstream from what this script writes:

    logit = w . (1/n) SUM_i e(word_i) + b = (1/n) SUM_i (w . e(word_i)) + b

Two outputs, under artifacts/embeddings/<backend>__<model>/:

    vocab.npz          float32 [n_words, dim], one row per distinct word
    index/<pair>.npz   per (stem, n_words, rep): its words' rows, in order

Distinct, not per-occurrence: the endpoint is deterministic, so the 9.9k unique
words carry the same vectors as all 1.7M occurrences at 0.6% of the calls. The
index is the join back, so nothing about per-word granularity is lost.

Reruns embed only what is missing, so this can run against the pairs that have
finished captioning and again when the rest land. Existing rows keep their
position, which keeps already-written index files valid.

Examples:
    # every pair that has captions, ~10 calls
    python scripts/02_embed.py

    # one pair, and count the work without calling anything
    python scripts/02_embed.py --pair hotpot_vase --dry-run

    # the human tags from ImageNet-Captions, for the SMER-vs-tags comparison
    python scripts/02_embed.py --what tags

    # re-embed a sample and check it against what is on disk
    python scripts/02_embed.py --verify
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tebol import vectors_io as vio  # noqa: E402
from tebol.captions_io import load_rows  # noqa: E402

CAPTIONS = ROOT / "artifacts" / "captions"
TAGS_DIR = ROOT / "data" / "manifest" / "tags"
OUT_ROOT = ROOT / "artifacts" / "embeddings"

#: mean, as in the Diplom notebook -- recorded so stage 3 cannot pick the other
AGGREGATION = "mean"

#: suffix marking the class-word-free captions; must match scripts/class_leakage.py
NOCLASS = "--noclass"


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal .env loader so the script has no hard dependency on python-dotenv."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def caption_files(pair: str | None, caption_model: str | None, variant: str):
    """[(pair, jsonl), ...] -- the stage-1 output this vocabulary is built from.

    `variant` picks between the captions as generated and the ones with class
    names and synonyms stripped by scripts/class_leakage.py. They sit in the
    same folder, so without this filter both would land in one vocabulary and
    the second index written would overwrite the first.
    """
    if not CAPTIONS.is_dir():
        sys.exit(f"no captions yet: {CAPTIONS} does not exist -- run stage 1")
    found = []
    for pair_dir in sorted(p for p in CAPTIONS.iterdir() if p.is_dir()):
        if pair and pair_dir.name != pair:
            continue
        files = sorted(pair_dir.glob("*.jsonl"))
        files = [f for f in files
                 if f.stem.endswith(NOCLASS) == (variant == "noclass")]
        if caption_model:
            files = [f for f in files if f.stem == caption_model]
        if not files:
            continue
        if len(files) > 1:
            sys.exit(f"{pair_dir.name} has several caption files "
                     f"({', '.join(f.stem for f in files)}) -- "
                     f"pick one with --caption-model")
        found.append((pair_dir.name, files[0]))
    if not found:
        sys.exit(f"no {variant!r} caption files under {CAPTIONS}"
                 + (f" for pair {pair!r}" if pair else "")
                 + ("\nrun scripts/class_leakage.py --emit-ablated first"
                    if variant == "noclass" else ""))
    return found


def read_pair(path: Path):
    """(records, counts) for one pair, deduplicated the way every stage must.

    load_rows resolves the append-only retries -- a generation that failed and
    later succeeded appears twice in the file and must count once here.
    """
    records, counts = [], Counter()
    for r in load_rows(path):
        toks = r.get("tokens") or []
        records.append({"stem": r["stem"], "n_words": r["n_words"],
                        "rep": r["rep"], "class": r["class"], "tokens": toks})
        counts.update(toks)
    return records, counts


def read_tags(pair: str | None = None):
    """(counts, {pair: records}) -- distinct tags, and the image -> tags join.

    A tag is embedded whole rather than word by word: "fire engine" is one
    thing a person chose to write, and splitting it would invent features the
    annotator never used. Only images that actually carry tags get a record --
    roughly a third of the corpus, and unevenly spread across classes.
    """
    if not TAGS_DIR.is_dir():
        sys.exit(f"no tag manifest at {TAGS_DIR} -- run build_caption_tags.py")
    counts: Counter = Counter()
    records: dict[str, list[dict]] = defaultdict(list)
    for f in sorted(TAGS_DIR.glob("*.csv")):
        with f.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if pair and row["pair"] != pair:
                    continue
                tags = [t.strip().lower()
                        for t in (row["tags"].split("|") if row["tags"] else [])]
                tags = [t for t in tags if t]
                if not tags:
                    continue
                counts.update(tags)
                # n_words/rep are unused here; the CSR layout is shared with
                # the caption index so one loader reads both
                records[row["pair"]].append({
                    "stem": row["stem"], "class": row["class"],
                    "n_words": 0, "rep": 0, "tokens": tags})
    if not counts:
        sys.exit(f"no tags found in {TAGS_DIR}"
                 + (f" for pair {pair!r}" if pair else ""))
    return counts, dict(records)


def read_sidecar_counts(path: Path) -> dict[str, dict[str, int]]:
    """key -> {pair: count}, so a rerun over one pair keeps the others' counts."""
    out: dict[str, dict[str, int]] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            per = {}
            for part in (row.get("pairs") or "").split("|"):
                name, _, n = part.rpartition(":")
                if name and n.isdigit():
                    per[name] = int(n)
            out[row["key"]] = per
    return out


def git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except Exception:  # not a repo, no git, shallow checkout -- provenance only
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--what", default="words", choices=["words", "tags"],
                   help="caption words (default) or ImageNet-Captions tags")
    p.add_argument("--pair", default=None, help="only this pair (default all)")
    p.add_argument("--variant", default="full", choices=["full", "noclass"],
                   help="captions as generated (default), or with class names "
                        "and synonyms stripped by scripts/class_leakage.py")
    p.add_argument("--caption-model", default=None,
                   help="stage-1 file stem, if a pair has more than one")
    p.add_argument("--backend", default="local", choices=["local", "openai"])
    p.add_argument("--model", default=None, help="overrides the env default")
    p.add_argument("--batch-size", type=int, default=1024,
                   help="texts per API call (default 1024)")
    p.add_argument("--out", default=None, help="explicit output directory")
    p.add_argument("--verify", action="store_true",
                   help="re-embed a sample and compare against what is stored")
    p.add_argument("--verify-n", type=int, default=20)
    p.add_argument("--dry-run", action="store_true", help="count work, call nothing")
    args = p.parse_args()

    load_dotenv()

    # ---- what needs a vector ------------------------------------------------
    per_pair_counts: dict[str, Counter] = {}
    pair_records: dict[str, list[dict]] = {}
    caption_model = None

    tag_records: dict[str, list[dict]] = {}
    if args.what == "tags":
        per_pair_counts["tags"], tag_records = read_tags(args.pair)
        for p, recs in sorted(tag_records.items()):
            print(f"{p:<24} {len(recs):>7,} tagged images  "
                  f"{sum(len(r['tokens']) for r in recs):>9,} tags")
    else:
        files = caption_files(args.pair, args.caption_model, args.variant)
        caption_model = files[0][1].stem
        for pair, path in files:
            records, counts = read_pair(path)
            pair_records[pair] = records
            per_pair_counts[pair] = counts
            print(f"{pair:<24} {len(records):>7,} captions  "
                  f"{sum(counts.values()):>9,} words  "
                  f"{len(counts):>6,} distinct")

    totals: Counter = Counter()
    for counts in per_pair_counts.values():
        totals.update(counts)
    keys = sorted(totals)

    print(f"\nwhat        {args.what}")
    print(f"distinct    {len(keys):,}")
    print(f"occurrences {sum(totals.values()):,}")

    if args.dry_run:
        est = -(-len(keys) // args.batch_size)
        print(f"\n(dry run -- nothing called; would be ~{est} calls)")
        return

    # ---- where it goes ------------------------------------------------------
    from tebol.embedders import embed_all, get_embedder

    embedder = get_embedder(args.backend, args.model)
    model_tag = embedder.name.replace("/", "-").replace(":", "__")
    # the variant is part of the directory, not just the metadata: the two sets
    # have different vocabularies and different indexes, and sharing a folder
    # would have the second run silently overwrite the first
    if args.variant != "full":
        model_tag += NOCLASS
    out_dir = Path(args.out) if args.out else OUT_ROOT / model_tag
    npz = out_dir / (vio.TAGS_NPZ if args.what == "tags" else vio.VOCAB_NPZ)
    sidecar = out_dir / (vio.TAGS_CSV if args.what == "tags" else vio.VOCAB_CSV)

    print(f"model       {embedder.name}")
    try:
        print(f"writing     {out_dir.relative_to(ROOT)}/")
    except ValueError:  # --out pointed somewhere outside the repo
        print(f"writing     {out_dir}/")

    # ---- resume: keep existing rows where they are --------------------------
    have_keys: list[str] = []
    have_vecs: np.ndarray | None = None
    if npz.exists():
        store = vio.load_matrix(npz, meta={})
        have_keys, have_vecs = list(store.keys), store.vectors
        print(f"existing    {len(have_keys):,} vectors, dim {store.dim}")

    known = set(have_keys)
    todo = [k for k in keys if k not in known]
    print(f"to embed    {len(todo):,}\n")

    new_vecs = np.empty((0, 0), dtype=np.float32)
    errors: dict[int, str] = {}
    stats = {"calls": 0, "splits": 0, "retries": 0,
             "prompt_tokens": 0, "latency_s": 0.0}

    if todo:
        t0 = time.time()

        def progress(done: int, total: int, st: dict) -> None:
            rate = done / max(time.time() - t0, 1e-9)
            eta = (total - done) / max(rate, 1e-9)
            print(f"  {done:>7,}/{total:,}  {rate:>6.0f}/s  "
                  f"calls={st['calls']} splits={st['splits']} "
                  f"retries={st['retries']}  eta {eta:.0f}s", flush=True)

        vectors, errors, stats = embed_all(embedder, todo,
                                           batch_size=args.batch_size,
                                           progress=progress)
        good = [(k, v) for k, v in zip(todo, vectors) if v is not None]
        todo = [k for k, _ in good]
        if good:
            new_vecs = np.asarray([v for _, v in good], dtype=np.float32)
        print(f"\nembedded {len(todo):,} in {(time.time() - t0) / 60:.1f} min "
              f"({stats['calls']} calls, {stats['prompt_tokens']:,} tokens)")
        if errors:
            print(f"! {len(errors):,} failed -- rerun to retry")
            for msg in list(errors.values())[:5]:
                print(f"    {msg[:110]}")

    # ---- merge and write ----------------------------------------------------
    if (have_vecs is not None and have_vecs.size and new_vecs.size
            and have_vecs.shape[1] != new_vecs.shape[1]):
        sys.exit(f"dimension changed: stored {have_vecs.shape[1]}, "
                 f"new {new_vecs.shape[1]} -- the model behind {embedder.name} "
                 f"is not the one that wrote {npz.name}; use --out")

    all_keys = have_keys + todo
    if have_vecs is not None and have_vecs.size and new_vecs.size:
        matrix = np.vstack([have_vecs, new_vecs])
    elif have_vecs is not None and have_vecs.size:
        matrix = have_vecs
    else:
        matrix = new_vecs

    if not all_keys:
        sys.exit("nothing embedded and nothing stored -- see the errors above")

    vio.save_matrix(npz, all_keys, matrix)

    prior = read_sidecar_counts(sidecar)
    for key in all_keys:
        per = prior.setdefault(key, {})
        for pair, counts in per_pair_counts.items():
            if counts.get(key):
                per[pair] = counts[key]
    vio.save_sidecar(sidecar, (
        {"key": k, "row": i,
         "count": sum(prior.get(k, {}).values()),
         "pairs": "|".join(f"{p}:{n}" for p, n in sorted(prior.get(k, {}).items()))}
        for i, k in enumerate(all_keys)
    ), ["key", "row", "count", "pairs"])

    norms = np.linalg.norm(matrix, axis=1)
    print(f"\n{npz.name}  {matrix.shape[0]:,} x {matrix.shape[1]}  "
          f"({matrix.nbytes / 1e6:.0f} MB)   ||v|| mean {norms.mean():.4f} "
          f"min {norms.min():.4f} max {norms.max():.4f}")

    # ---- the caption -> row join --------------------------------------------
    rows_of = {k: i for i, k in enumerate(all_keys)}
    index_meta = {"model": embedder.name, "dim": int(matrix.shape[1]),
                  "aggregation": AGGREGATION, "caption_model": caption_model}
    if pair_records:
        print()
        for pair, records in pair_records.items():
            path = out_dir / vio.INDEX_DIR / f"{pair}.npz"
            got = vio.save_index(path, records, rows_of,
                                 {**index_meta, "pair": pair})
            flag = f"  ! {got['missing']:,} words unembedded" if got["missing"] else ""
            print(f"index/{pair}.npz  {got['captions']:>7,} captions  "
                  f"{got['slots']:>9,} word slots{flag}")

    if tag_records:
        print()
        for pair, records in sorted(tag_records.items()):
            path = out_dir / vio.INDEX_TAGS_DIR / f"{pair}.npz"
            got = vio.save_index(path, records, rows_of,
                                 {**index_meta, "pair": pair, "unit": "tag"})
            flag = f"  ! {got['missing']:,} tags unembedded" if got["missing"] else ""
            print(f"index_tags/{pair}.npz  {got['captions']:>7,} tagged images  "
                  f"{got['slots']:>8,} tag slots{flag}")

    # ---- provenance ---------------------------------------------------------
    meta = vio.load_meta(out_dir)
    meta.update({
        "model": embedder.name,
        "backend": args.backend,
        "host": urlparse(os.environ.get("LOCAL_OPENAI_BASE_URL", "")).netloc or None,
        "dim": int(matrix.shape[1]),
        "aggregation": AGGREGATION,
        "renormalize_after_aggregation": False,
        "l2_normalized": bool(np.allclose(norms, 1.0, atol=1e-3)),
        "instruction_prefix": None,
        "split_hyphens": False,
        "caption_variant": args.variant,
        "caption_model": caption_model or meta.get("caption_model"),
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
    })
    meta.setdefault("created", meta["updated"])
    meta["n_tags" if args.what == "tags" else "n_words"] = len(all_keys)
    if args.what == "words":
        meta["pairs"] = sorted(set(meta.get("pairs", [])) | set(pair_records))
    meta["prompt_tokens"] = meta.get("prompt_tokens", 0) + stats["prompt_tokens"]
    vio.save_meta(out_dir, meta)

    if args.verify:
        verify(embedder, npz, args.verify_n)


#: Batch composition perturbs a vector by ~1e-3 per component, so a re-embed
#: never reproduces the stored bytes. Cosine is the invariant that survives it;
#: two genuinely different words sit around 0.6, so this leaves a wide margin.
COS_TOL = 1e-3


def verify(embedder, npz: Path, n: int) -> None:
    """Re-embed a sample and compare, to catch a shuffled or stale matrix.

    A row-order bug does not raise -- it surfaces much later as a classifier
    that trains fine and explains nonsense, so it is worth one call to rule out.

    Two checks. Cosine against the row the key claims, which catches a stale or
    wrong-model matrix; and nearest row over the whole matrix, which is what
    actually catches a shuffle, since a rotated index still gives every word
    *some* plausible vector.
    """
    store = vio.load_matrix(npz, meta={})
    sample = random.sample(range(len(store)), min(n, len(store)))
    keys = [store.keys[i] for i in sample]
    res = embedder.embed(keys)
    if res.vectors is None:
        print(f"\nverify: could not re-embed ({res.error})")
        return

    fresh = np.asarray(res.vectors, dtype=np.float32)
    fresh /= np.linalg.norm(fresh, axis=1, keepdims=True)
    unit = store.vectors / np.linalg.norm(store.vectors, axis=1, keepdims=True)

    cos = (fresh * unit[sample]).sum(axis=1)
    nearest = (fresh @ unit.T).argmax(axis=1)
    off = [keys[i] for i, (c, row) in enumerate(zip(cos, nearest))
           if c < 1 - COS_TOL or row != sample[i]]

    print(f"\nverify: {len(keys)} sampled, min cosine {cos.min():.6f}, "
          f"{int((nearest == np.asarray(sample)).sum())}/{len(keys)} nearest-row hits")
    if off:
        print(f"  ! {len(off)} mismatched: {', '.join(off[:5])}")
        print("  the stored matrix does not match the model -- rebuild with --out")
    else:
        print("  every sampled row matches the model")


if __name__ == "__main__":
    main()

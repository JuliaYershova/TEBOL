#!/usr/bin/env python3
"""Turn the saved GPT-4o configuration into the stage-3 inputs.

The GPT-4o run predates this repository. What survives of it, per pair, is one
description per image and, for every word of that description, the
text-embedding-ada-002 vector the word was embedded with
(`test_package_data*/embedded_dataset.csv`, columns Image / Description /
Embedding / Label). Nothing is generated or embedded again here: the stored
descriptions and the stored vectors are re-used as they are.

Two things are rebuilt so the corrected pipeline can read them:

    vocab.npz        one vector per distinct token. The old run embedded every
                     occurrence separately and the API is not bit-stable, so a
                     token that occurs many times has many near-identical
                     vectors (cosine to their mean: median 1.000000, worst
                     0.995 on the sample checked). They are averaged and the
                     mean is renormalised to unit length.
    index/<pair>.npz which vocab rows each description uses, in order.

The tokens are the whitespace tokens the old run embedded ("by." stays "by."),
because the vector belongs to that exact string. Descriptions have a single
length setting, 7, and a single repetition.

Class labels are taken from the Qwen3-VL index of the same image, so both
configurations name a class identically and share its folds. The image sets
are identical (checked: 2,866 of 2,866 for ambulance / firetruck).

    python scripts/build_gpt4o_inputs.py
    python scripts/build_gpt4o_inputs.py --pair ant_bee
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tebol import arms as arms_mod  # noqa: E402
from tebol import vectors_io as vio  # noqa: E402

#: where the old run left its per-pair data
SRC = Path("/Users/kjzg289/Library/CloudStorage/OneDrive-AZCollaboration/"
           "study/2 semestr/Diplom/data")
PAIR_DIR = {"ambulance_firetruck": "test_package_data1",
            "acousticguitar_violin": "test_package_data2",
            "ant_bee": "test_package_data3",
            "hotpot_vase": "test_package_data4",
            "cucumber_zucchini": "test_package_data5"}
SETUP = 7
MODEL = "openai:text-embedding-ada-002"


def read_descriptions(pair: str, stem_to_class: dict[str, str]):
    """Stored descriptions only, for a pair whose stored vectors are not ada-002."""
    path = SRC / PAIR_DIR[pair] / "embedded_dataset.csv"
    csv.field_size_limit(10 ** 9)
    records = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            stem = Path(row["Image"]).stem
            if stem not in stem_to_class:
                raise SystemExit(f"{pair} {stem}: not in the Qwen3-VL image set")
            records.append({"stem": stem, "n_words": SETUP, "rep": 0,
                            "class": stem_to_class[stem],
                            "tokens": row["Description"].split()})
    return records


def _post_embeddings(chunk: list[str], key: str) -> list[dict]:
    """POST one batch with curl, which verifies the server certificate normally.

    This network's TLS proxy presents a CA certificate that Python 3.13's
    default strict check rejects while curl accepts it, so Python's own HTTP
    clients cannot connect. The key goes to curl on stdin, not on the command
    line and not into a file.
    """
    import json
    import subprocess
    import tempfile

    body = json.dumps({"model": "text-embedding-ada-002", "input": chunk})
    config = (f'url = "https://api.openai.com/v1/embeddings"\n'
              f'header = "Authorization: Bearer {key}"\n'
              'header = "Content-Type: application/json"\n')
    for attempt in range(3):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as fh:
            fh.write(body)
            fh.flush()
            out = subprocess.run(
                ["curl", "-s", "-m", "120", "--config", "-",
                 "--data-binary", f"@{fh.name}"],
                input=config, capture_output=True, text=True)
        try:
            resp = json.loads(out.stdout)
        except json.JSONDecodeError:
            resp = {}
        if "data" in resp:
            return resp["data"]
        err = resp.get("error", {}).get("message", out.stderr[:200] or "no reply")
        print(f"    request failed ({err[:120]}); attempt {attempt + 1}/3",
              flush=True)
    raise SystemExit("the embeddings request failed three times")


def embed_with_api(tokens: list[str], cache: Path) -> dict[str, np.ndarray]:
    """text-embedding-ada-002 for tokens that have no stored vector.

    Each token is embedded on its own, as the original run did. The key is read
    from the OPENAI_API_KEY environment variable and is never written anywhere;
    vectors already fetched are kept in `cache` so a rerun calls nothing.
    """
    import os

    done: dict[str, np.ndarray] = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        done = dict(zip(map(str, z["keys"]), z["vectors"]))
    todo = [t for t in tokens if t not in done]
    if todo:
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit(f"{len(todo)} tokens need an ada-002 vector: "
                             "set OPENAI_API_KEY for this run")
        for i in range(0, len(todo), 256):
            chunk = todo[i:i + 256]
            data = _post_embeddings(chunk, os.environ["OPENAI_API_KEY"])
            for tok, item in zip(chunk, sorted(data, key=lambda d: d["index"])):
                done[tok] = np.asarray(item["embedding"], dtype=np.float64)
            print(f"    embedded {min(i + 256, len(todo))}/{len(todo)} tokens",
                  flush=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, keys=np.asarray(list(done), dtype=object),
                            vectors=np.stack(list(done.values())))
    return done


class Unsupported(Exception):
    """The stored file does not hold ada-002 word vectors."""


def read_pair(pair: str, stem_to_class: dict[str, str]):
    """Stream one pair's file. Returns records and per-token (sum, count)."""
    path = SRC / PAIR_DIR[pair] / "embedded_dataset.csv"
    csv.field_size_limit(10 ** 9)
    records, acc = [], {}
    t0 = time.time()
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if "Embedding" not in (reader.fieldnames or []):
            raise Unsupported(
                f"{pair}: {path.name} has columns {reader.fieldnames}, with no "
                "'Embedding' matrix of ada-002 word vectors (cucumber_zucchini "
                "stores 300-dimensional vectors from an earlier embedding model)")
        for i, row in enumerate(reader, 1):
            stem = Path(row["Image"]).stem
            tokens = row["Description"].split()
            mat = np.asarray(json.loads(row["Embedding"]), dtype=np.float64)
            if mat.shape[0] != len(tokens):
                raise SystemExit(f"{pair} {stem}: {mat.shape[0]} vectors for "
                                 f"{len(tokens)} tokens")
            if stem not in stem_to_class:
                raise SystemExit(f"{pair} {stem}: not in the Qwen3-VL image set")
            for tok, vec in zip(tokens, mat):
                if tok in acc:
                    acc[tok][0] += vec
                    acc[tok][1] += 1
                else:
                    acc[tok] = [vec.copy(), 1]
            records.append({"stem": stem, "n_words": SETUP, "rep": 0,
                            "class": stem_to_class[stem], "tokens": tokens})
            if i % 500 == 0:
                print(f"    {pair}: {i} descriptions, {time.time() - t0:.0f}s",
                      flush=True)
    return records, acc


def read_pair_cached(pair: str, stem_to_class: dict[str, str], cache_dir: Path,
                     retries: int = 40):
    """read_pair, remembered per pair and retried while OneDrive downloads.

    The source files are cloud-only until first read, and a read that outruns
    the download raises a timeout. Each retry reads further into a file that is
    by then partly local, so it converges; a finished pair is pickled so a
    later run never reads it again.
    """
    cache = cache_dir / f"{pair}.pkl"
    if cache.exists():
        print(f"  {pair}: using {cache.name}", flush=True)
        with cache.open("rb") as fh:
            return pickle.load(fh)
    for attempt in range(1, retries + 1):
        try:
            out = read_pair(pair, stem_to_class)
            break
        except Unsupported:
            raise
        except (TimeoutError, OSError) as exc:
            print(f"  {pair}: read failed ({exc}); attempt {attempt}/{retries}, "
                  f"waiting for the download", flush=True)
            time.sleep(30)
    else:
        raise SystemExit(f"{pair}: could not read the source file")
    cache_dir.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as fh:
        pickle.dump(out, fh)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", action="append", choices=sorted(PAIR_DIR))
    ap.add_argument("--embed-missing", action="store_true",
                    help="for a pair whose stored vectors are not ada-002, take the "
                         "stored descriptions and embed only their tokens that have "
                         "no stored vector, with the OpenAI API (OPENAI_API_KEY)")
    args = ap.parse_args()
    pairs = args.pair or sorted(PAIR_DIR)

    out_dir = ROOT / "artifacts" / "embeddings" / arms_mod.EMB_GPT4O
    qwen = ROOT / "artifacts" / "embeddings" / arms_mod.EMB_FULL

    all_acc, all_records, skipped, pending = {}, {}, [], []
    for pair in pairs:
        ix = vio.load_index(qwen / vio.INDEX_DIR / f"{pair}.npz")
        stem_to_class = dict(zip(map(str, ix.stem), map(str, ix.cls)))
        print(f"{pair}: {len(stem_to_class)} images in the Qwen3-VL set", flush=True)
        try:
            records, acc = read_pair_cached(pair, stem_to_class,
                                            out_dir / "_partial")
        except Unsupported as exc:
            if not args.embed_missing:
                print(f"  SKIPPED  {exc}", flush=True)
                skipped.append(pair)
                continue
            records = read_descriptions(pair, stem_to_class)
            toks = sorted({t for r in records for t in r["tokens"]})
            pending.append((pair, records, toks))
            all_records[pair] = records
            print(f"  {pair}: {len(records)} stored descriptions, {len(toks)} "
                  "distinct tokens; stored vectors are not ada-002", flush=True)
            continue
        missing = set(stem_to_class) - {r["stem"] for r in records}
        print(f"  {len(records)} descriptions, {len(acc)} distinct tokens, "
              f"{len(missing)} Qwen3-VL images without a GPT-4o description")
        all_records[pair] = records
        for tok, (s, n) in acc.items():
            if tok in all_acc:
                all_acc[tok][0] += s
                all_acc[tok][1] += n
            else:
                all_acc[tok] = [s, n]

    embedded_now = []
    for pair, records, toks in pending:
        have = [t for t in toks if t in all_acc]
        need = [t for t in toks if t not in all_acc]
        print(f"{pair}: {len(have)} tokens reuse a stored ada-002 vector, "
              f"{len(need)} need the API", flush=True)
        fresh = embed_with_api(need, out_dir / "_partial" / f"{pair}_api.npz")
        for tok, vec in fresh.items():
            all_acc[tok] = [vec.copy(), 1]
        embedded_now.append({"pair": pair, "tokens_from_api": len(need),
                             "tokens_reusing_stored": len(have)})

    keys = sorted(all_acc)
    vecs = np.stack([all_acc[k][0] / all_acc[k][1] for k in keys])
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    matrix = vecs.astype(np.float32)
    rows_of = {k: i for i, k in enumerate(keys)}

    vio.save_matrix(out_dir / vio.VOCAB_NPZ, keys, matrix)
    vio.save_sidecar(out_dir / vio.VOCAB_CSV,
                     ({"key": k, "row": i, "count": all_acc[k][1]}
                      for i, k in enumerate(keys)), ["key", "row", "count"])
    meta = {"model": MODEL, "dim": int(matrix.shape[1]), "aggregation": "mean",
            "l2_normalized": True, "caption_model": "gpt-4o",
            "source": "stored descriptions and stored per-word ada-002 vectors "
                      "of the original run; occurrences of a token averaged",
            "setup": SETUP, "embedded_now": embedded_now}
    for pair, records in all_records.items():
        got = vio.save_index(out_dir / vio.INDEX_DIR / f"{pair}.npz", records,
                             rows_of, {**meta, "pair": pair})
        print(f"index/{pair}.npz  {got['captions']:>6,} descriptions  "
              f"{got['slots']:>7,} word slots  missing {got['missing']}")
    vio.save_meta(out_dir, meta)
    print(f"\nvocab {matrix.shape[0]:,} x {matrix.shape[1]} -> {out_dir}")
    if skipped:
        print(f"skipped pairs: {', '.join(skipped)}")


if __name__ == "__main__":
    main()

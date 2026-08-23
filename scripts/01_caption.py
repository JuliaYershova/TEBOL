#!/usr/bin/env python3
"""Stage 1 -- generate image captions.

For every image, generate a caption at each word count (3/5/7/10) and repeat
each one N times, so caption stability can be measured afterwards.

The prompt is the Diplom one, a single user message with no system
instruction, with only the number changing:

    Describe this image in {n} words.

Output is append-only JSONL, one row per (image, n_words, rep):

    artifacts/captions/<pair>/<backend>__<model>.jsonl

Reruns skip work already in the file, so a job killed by a walltime limit
resumes where it stopped instead of starting over.

Examples:
    # smoke test: 5 images, all word counts, 5 reps  -> 125 generations
    python scripts/01_caption.py --pair cucumber_zucchini --backend local --limit 5

    # full pair on the school server
    python scripts/01_caption.py --pair cucumber_zucchini --backend local

    # count the work without calling any model
    python scripts/01_caption.py --pair cucumber_zucchini --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tebol.text import tokenize  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT_ROOT = ROOT / "artifacts" / "captions"
IMG_EXT = {".jpeg", ".jpg", ".png"}
DEFAULT_WORDS = [3, 5, 7, 10]


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


def iter_images(pair: str, limit: int | None = None):
    """(stem, filename, class, wnid, path) for one pair, deterministic order."""
    pair_dir = RAW / pair
    if not pair_dir.is_dir():
        sys.exit(f"no such pair: {pair_dir}")
    items = []
    for class_dir in sorted(c for c in pair_dir.iterdir() if c.is_dir()):
        files = sorted(f for f in class_dir.iterdir() if f.suffix.lower() in IMG_EXT)
        if limit:  # take from each class so both are represented
            files = files[:limit]
        for f in files:
            items.append((f.stem, f.name, class_dir.name, f.stem.split("_")[0], f))
    return items


def load_done(path: Path) -> set[tuple[str, int, int]]:
    """Keys already present, so we never pay for the same generation twice."""
    done: set[tuple[str, int, int]] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a torn last line from a killed job
            if r.get("caption"):  # failed rows are retried on the next run
                done.add((r["stem"], r["n_words"], r["rep"]))
    return done


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pair", required=True, help="folder name under data/raw/")
    p.add_argument("--backend", default="local", choices=["local", "openai"])
    p.add_argument("--model", default=None, help="overrides the env default")
    p.add_argument("--words", default=",".join(map(str, DEFAULT_WORDS)),
                   help="comma-separated word counts (default 3,5,7,10)")
    p.add_argument("--reps", type=int, default=5,
                   help="generations per image per word count (default 5)")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="must be > 0 or all reps are identical (default 1.0)")
    p.add_argument("--seed", type=int, default=None,
                   help="base seed; rep index is added to it")
    p.add_argument("--limit", type=int, default=None, help="first N images per class")
    p.add_argument("--out", default=None, help="explicit jsonl path")
    p.add_argument("--split-hyphens", action="store_true",
                   help="'red-figure' -> 'red','figure' (default keeps it whole)")
    p.add_argument("--dry-run", action="store_true", help="count work, call nothing")
    args = p.parse_args()

    words = [int(w) for w in args.words.split(",") if w.strip()]
    if args.temperature <= 0 and args.reps > 1:
        print("! temperature=0 with reps>1: every repetition will be identical",
              file=sys.stderr)

    load_dotenv()
    images = iter_images(args.pair, args.limit)
    if not images:
        sys.exit(f"no images under {RAW / args.pair}")

    total = len(images) * len(words) * args.reps
    by_class = Counter(c for _, _, c, _, _ in images)
    print(f"pair       {args.pair}")
    print(f"images     {len(images)}  ({dict(by_class)})")
    print(f"words      {words}")
    print(f"reps       {args.reps}   temperature {args.temperature}")
    print(f"total      {total:,} generations")

    if args.dry_run:
        print("\n(dry run -- nothing called)")
        return

    captioner = None if args.dry_run else _build(args)
    model_tag = captioner.name.replace("/", "-").replace(":", "__")
    out = Path(args.out) if args.out else OUT_ROOT / args.pair / f"{model_tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    done = load_done(out)
    todo = [(im, n, r) for im in images for n in words for r in range(args.reps)
            if (im[0], n, r) not in done]
    print(f"already done {len(done):,} -- {len(todo):,} to generate")
    try:
        shown = out.relative_to(ROOT)
    except ValueError:  # --out pointed somewhere outside the repo
        shown = out
    print(f"writing      {shown}\n")
    if not todo:
        print("nothing to do")
        return

    ok = fail = 0
    latencies: list[float] = []
    t0 = time.time()
    with out.open("a", encoding="utf-8") as fh:
        for i, ((stem, fname, cls, wnid, path), n_words, rep) in enumerate(todo, 1):
            seed = None if args.seed is None else args.seed + rep
            t_row = time.perf_counter()
            res = captioner.caption(path, n_words, args.temperature, seed)
            elapsed = round(time.perf_counter() - t_row, 3)
            toks = tokenize(res.text, args.split_hyphens) if res.text else []
            row = {
                "stem": stem, "image": fname, "class": cls, "wnid": wnid,
                "pair": args.pair, "n_words": n_words, "rep": rep,
                "caption": res.text,
                "caption_norm": " ".join(toks),
                "tokens": toks, "n_tokens": len(toks),
                "split_hyphens": args.split_hyphens,
                "error": res.error,
                "finish_reason": res.finish_reason,
                "model": captioner.name, "backend": args.backend,
                "temperature": args.temperature, "seed": seed,
                "prompt_tokens": res.prompt_tokens,
                "completion_tokens": res.completion_tokens,
                # latency_s = the API call; elapsed_s also covers retry backoff
                "latency_s": res.latency_s,
                "elapsed_s": elapsed,
                "attempts": res.attempts,
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()  # a killed job keeps everything up to this point

            ok, fail = (ok + 1, fail) if res.text else (ok, fail + 1)
            latencies.append(elapsed)
            if i % 25 == 0 or i == len(todo):
                rate = i / max(time.time() - t0, 1e-9)
                eta = (len(todo) - i) / max(rate, 1e-9)
                print(f"  {i:>7,}/{len(todo):,}  ok={ok:,} fail={fail:,}  "
                      f"{rate:.2f}/s  {statistics.mean(latencies[-25:]):.1f}s/gen  "
                      f"eta {eta / 60:.0f}m", flush=True)

    print(f"\ndone: {ok:,} captions, {fail:,} failures in "
          f"{(time.time() - t0) / 60:.1f} min")
    if latencies:
        s = sorted(latencies)
        p95 = s[min(len(s) - 1, int(0.95 * len(s)))]
        print(f"per generation: mean {statistics.mean(s):.2f}s  "
              f"median {statistics.median(s):.2f}s  "
              f"p95 {p95:.2f}s  min {s[0]:.2f}s  max {s[-1]:.2f}s")
    if fail:
        print("rerun the same command to retry the failures")


def _build(args):
    from tebol.captioners import get_captioner
    return get_captioner(args.backend, args.model)


if __name__ == "__main__":
    main()

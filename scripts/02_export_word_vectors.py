#!/usr/bin/env python3
"""Stage 2b -- lay the word vectors out by setup and by image.

Stage 2 stores one row per distinct word in vocab.npz and one compact index per
pair. That is the right shape for storage and the wrong shape for working: SMER
trains one model per (pair, word count), and inspecting a single image means
digging a stem out of a 66k-row index. This writes both views.

    by_setup/<pair>/w03.npz              every caption at 3 words, that pair
    by_image/<pair>/<class>/<stem>.npz   one image, all its setups

Both hold vocab *rows*, not vectors. A word's vector lives once in vocab.npz;
materialising it per occurrence would turn ~50 MB of ids into ~29 GB of floats
for no new information. Reading is a lookup away:

    store = vio.load_vocab(D)
    img   = vio.load_image_vectors(f"{D}/by_image/hotpot_vase/hotpot/n07590611_1.npz")
    words, vectors = img.get(n_words=5, rep=0, store=store)   # (6,) and (6, 2560)

No API calls -- this is a reshape of what stage 2 already wrote, so it is safe
to rerun at any time and takes seconds.

Examples:
    python scripts/02_export_word_vectors.py
    python scripts/02_export_word_vectors.py --pair hotpot_vase
    python scripts/02_export_word_vectors.py --layout setup --dry-run
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tebol import vectors_io as vio  # noqa: E402

OUT_ROOT = ROOT / "artifacts" / "embeddings"


def pick_run(explicit: str | None) -> Path:
    """The stage-2 output directory to reshape."""
    if explicit:
        d = Path(explicit)
        if not (d / vio.VOCAB_NPZ).exists():
            sys.exit(f"no {vio.VOCAB_NPZ} in {d}")
        return d
    runs = [d for d in sorted(OUT_ROOT.glob("*")) if (d / vio.VOCAB_NPZ).exists()]
    if not runs:
        sys.exit(f"no embeddings under {OUT_ROOT} -- run scripts/02_embed.py first")
    if len(runs) > 1:
        sys.exit("several embedding runs present "
                 f"({', '.join(d.name for d in runs)}) -- pick one with --dir")
    return runs[0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=None, help="stage-2 output dir (default: the only one)")
    p.add_argument("--pair", default=None, help="only this pair (default all)")
    p.add_argument("--layout", default="setup,image",
                   help="which views to write (default both)")
    p.add_argument("--dry-run", action="store_true", help="count work, write nothing")
    args = p.parse_args()

    run = pick_run(args.dir)
    layouts = {s.strip() for s in args.layout.split(",") if s.strip()}
    if unknown := layouts - {"setup", "image"}:
        sys.exit(f"unknown layout(s): {', '.join(sorted(unknown))} (setup | image)")

    store = vio.load_vocab(run)
    meta = vio.load_meta(run)
    print(f"run        {run.name}")
    print(f"vocab      {len(store):,} words x {store.dim}")
    print(f"aggregation{meta.get('aggregation', '?'):>6}   (recorded in meta.json)\n")

    indexes = sorted((run / vio.INDEX_DIR).glob("*.npz"))
    if args.pair:
        indexes = [p for p in indexes if p.stem == args.pair]
    if not indexes:
        sys.exit(f"no index files in {run / vio.INDEX_DIR}"
                 + (f" for pair {args.pair!r}" if args.pair else ""))

    grand = Counter()
    for path in indexes:
        pair = path.stem
        idx = vio.load_index(path)

        # ---- group once, use for both views ---------------------------------
        by_setup: dict[int, list[int]] = defaultdict(list)
        by_image: dict[str, list[int]] = defaultdict(list)
        for i in range(len(idx)):
            by_setup[int(idx.n_words[i])].append(i)
            by_image[str(idx.stem[i])].append(i)

        setups = sorted(by_setup)
        print(f"{pair}   {len(idx):,} captions   {len(by_image):,} images   "
              f"setups {setups}")

        if args.dry_run:
            grand.update(captions=len(idx), images=len(by_image))
            continue

        if "setup" in layouts:
            for n_words in setups:
                rows = by_setup[n_words]
                records = [{"stem": str(idx.stem[i]), "n_words": n_words,
                            "rep": int(idx.rep[i]), "class": str(idx.cls[i]),
                            "tokens": [store.keys[r] for r in idx.word_rows(i)]}
                           for i in rows]
                out = run / "by_setup" / pair / f"w{n_words:02d}.npz"
                got = vio.save_index(out, records,
                                     {k: i for i, k in enumerate(store.keys)},
                                     {**idx.meta, "pair": pair, "n_words": n_words})
                reps = sorted({int(idx.rep[i]) for i in rows})
                print(f"  by_setup/{pair}/w{n_words:02d}.npz  "
                      f"{got['captions']:>7,} captions  reps {reps}  "
                      f"{got['slots']:>8,} word slots")
                grand["setup_files"] += 1

        if "image" in layouts:
            for stem, rows in by_image.items():
                cls = str(idx.cls[rows[0]])
                captions = []
                for i in sorted(rows, key=lambda j: (idx.n_words[j], idx.rep[j])):
                    r = idx.word_rows(i, drop_missing=False)
                    captions.append({
                        "n_words": int(idx.n_words[i]), "rep": int(idx.rep[i]),
                        "rows": r,
                        "words": [store.keys[x] if x != vio.MISSING else ""
                                  for x in r],
                    })
                vio.save_image_vectors(
                    run / "by_image" / pair / cls / f"{stem}.npz",
                    stem, cls, pair, captions)
                grand["image_files"] += 1
            print(f"  by_image/{pair}/            {len(by_image):,} files")

        grand.update(captions=len(idx), images=len(by_image))

    print(f"\n{grand['captions']:,} captions over {grand['images']:,} images"
          + (f" -> {grand['setup_files']} setup files, "
             f"{grand['image_files']:,} image files" if not args.dry_run else
             "  (dry run -- nothing written)"))

    if not args.dry_run and "image" in layouts:
        show_example(run, store)


def show_example(run: Path, store: vio.VectorStore) -> None:
    """Print one image end to end, so the layout is obvious without reading code."""
    sample = next(iter(sorted((run / "by_image").rglob("*.npz"))), None)
    if sample is None:
        return
    img = vio.load_image_vectors(sample)
    print(f"\nexample  {sample.relative_to(run)}")
    print(f"  image {img.stem}  class {img.cls}  setups {img.setups()}")
    for n_words in img.setups():
        for rep, (words, vecs) in img.setup(n_words, store).items():
            print(f"    w{n_words:02d} rep{rep}  {str(vecs.shape):<12} "
                  f"{' '.join(words)}")
            break  # one rep per setup is enough to show the shape
    words, vecs = img.get(img.setups()[0], 0, store)
    mean = vecs.mean(axis=0)
    print(f"  caption vector = mean of {len(words)} word vectors -> {mean.shape}, "
          f"||v||={np.linalg.norm(mean):.4f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build per-class CSVs of ImageNet-Captions metadata (title / tags / description).

Reads:
    data/raw/<pair>/<class>/*.JPEG          images actually on disk
    data/annotations/<pair>/<class>/*.xml   ground-truth bboxes (optional)
    data/annotations/imagenet_captions.json ImageNet-Captions dump

Writes:
    data/manifest/tags/<pair>__<class>.csv  one row per image on disk

The synset id is read from the image filenames (n01234567_890.JPEG), so adding a
new pair needs no code change -- just drop the folders in and rerun.

Usage:
    python scripts/build_caption_tags.py                # all pairs
    python scripts/build_caption_tags.py hotpot_vase    # one pair
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
ANNOT = ROOT / "data" / "annotations"
CAPTIONS = ANNOT / "imagenet_captions.json"
OUT = ROOT / "data" / "manifest" / "tags"

IMG_EXT = {".jpeg", ".jpg", ".png"}
FIELDS = [
    "image", "stem", "wnid", "class", "pair",
    "has_caption", "has_bbox", "title", "tags", "n_tags", "description",
]


def discover_classes(only_pair: str | None = None):
    """Yield (pair, class_name, wnid, {stem: filename}) for every class folder."""
    for pair_dir in sorted(p for p in RAW.iterdir() if p.is_dir()):
        if only_pair and pair_dir.name != only_pair:
            continue
        for class_dir in sorted(c for c in pair_dir.iterdir() if c.is_dir()):
            images = {
                f.stem: f.name
                for f in class_dir.iterdir()
                if f.suffix.lower() in IMG_EXT
            }
            if not images:
                continue
            # synset id is the filename prefix; take the majority in case of strays
            prefixes = Counter(stem.split("_")[0] for stem in images)
            wnid, n = prefixes.most_common(1)[0]
            if n != len(images):
                strays = {p: c for p, c in prefixes.items() if p != wnid}
                print(f"  ! {pair_dir.name}/{class_dir.name}: mixed synsets, "
                      f"majority {wnid}, also {strays}")
            yield pair_dir.name, class_dir.name, wnid, images


def load_captions(wnids: set[str]) -> dict[str, dict]:
    """Load ImageNet-Captions, keeping only entries for the synsets we need."""
    if not CAPTIONS.exists():
        sys.exit(f"missing {CAPTIONS} -- download imagenet_captions.json first")
    print(f"loading {CAPTIONS.name} ({CAPTIONS.stat().st_size / 1e6:.0f} MB) ...")
    with CAPTIONS.open(encoding="utf-8") as fh:
        entries = json.load(fh)
    by_stem = {}
    for e in entries:
        if e.get("wnid") in wnids:
            by_stem[Path(e["filename"]).stem] = e
    print(f"  {len(entries):,} entries total, {len(by_stem):,} for our synsets")
    return by_stem


def bbox_stems(pair: str, class_name: str) -> set[str]:
    d = ANNOT / pair / class_name
    return {f.stem for f in d.rglob("*.xml")} if d.is_dir() else set()


def main() -> None:
    only_pair = sys.argv[1] if len(sys.argv) > 1 else None
    classes = list(discover_classes(only_pair))
    if not classes:
        sys.exit(f"no image folders found under {RAW}"
                 + (f" for pair {only_pair!r}" if only_pair else ""))

    captions = load_captions({wnid for _, _, wnid, _ in classes})
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"\n{'pair':<24} {'class':<16} {'images':>7} {'caption':>9} "
          f"{'tagged':>7} {'bbox':>6}")
    print("-" * 76)

    totals = Counter()
    written: set[Path] = set()
    for pair, class_name, wnid, images in classes:
        boxes = bbox_stems(pair, class_name)
        rows, n_cap, n_tagged, n_box = [], 0, 0, 0

        for stem in sorted(images):
            e = captions.get(stem)
            tags = e.get("tags") or [] if e else []
            has_box = stem in boxes
            n_cap += e is not None
            n_tagged += bool(tags)
            n_box += has_box
            rows.append({
                "image": images[stem],
                "stem": stem,
                "wnid": wnid,
                "class": class_name,
                "pair": pair,
                "has_caption": int(e is not None),
                "has_bbox": int(has_box),
                "title": (e.get("title") or "").replace("\n", " ") if e else "",
                # '|' because tags themselves contain commas and spaces
                "tags": "|".join(t.replace("|", "/") for t in tags),
                "n_tags": len(tags),
                "description": (e.get("description") or "").replace("\n", " ") if e else "",
            })

        path = OUT / f"{class_name}.csv"
        if path.exists() and path in written:
            print(f"  ! {path.name} already written by another pair -- overwriting")
        written.add(path)
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)

        n = len(rows)
        print(f"{pair:<24} {class_name:<16} {n:>7} "
              f"{n_cap:>4} {100 * n_cap / n:>4.0f}% {n_tagged:>7} {n_box:>6}")
        totals.update(images=n, captions=n_cap, tagged=n_tagged, boxes=n_box)

    print("-" * 76)
    print(f"{'TOTAL':<41} {totals['images']:>7} "
          f"{totals['captions']:>4} {100 * totals['captions'] / totals['images']:>4.0f}% "
          f"{totals['tagged']:>7} {totals['boxes']:>6}")
    print(f"\nwrote {len(classes)} csv files to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Report, per class pair, what data and ground truth we actually hold.

Counts images on disk and cross-references every ground-truth source we have:

    class label      folder name -> synset id            (always 100%)
    synset words     ImageNet words.txt                  (per class, not per image)
    bounding boxes   data/annotations/<pair>/<class>/*.xml
    captions + tags  ImageNet-Captions (title / tags / description)

Writes results/metrics/coverage.csv and prints a markdown table.

Usage:
    python scripts/coverage_report.py
"""

from __future__ import annotations

import csv
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
ANNOT = ROOT / "data" / "annotations"
CAPTIONS = ANNOT / "imagenet_captions.json"
WORDS_CACHE = ROOT / "data" / "manifest" / "synset_words.csv"
OUT_CSV = ROOT / "results" / "metrics" / "coverage.csv"
OUT_MD = ROOT / "results" / "metrics" / "coverage.md"

WORDS_URL = "https://image-net.org/data/words.txt"
IMG_EXT = {".jpeg", ".jpg", ".png"}


def discover():
    """[(pair, class, wnid, {stems}), ...] from the folder layout."""
    out = []
    for pair_dir in sorted(p for p in RAW.iterdir() if p.is_dir()):
        for class_dir in sorted(c for c in pair_dir.iterdir() if c.is_dir()):
            stems = {f.stem for f in class_dir.iterdir() if f.suffix.lower() in IMG_EXT}
            if not stems:
                continue
            wnid = Counter(s.split("_")[0] for s in stems).most_common(1)[0][0]
            out.append((pair_dir.name, class_dir.name, wnid, stems))
    return out


def synset_words(wnids: set[str]) -> dict[str, str]:
    """synset -> human words, cached locally so reruns need no network."""
    cache: dict[str, str] = {}
    if WORDS_CACHE.exists():
        with WORDS_CACHE.open(encoding="utf-8") as fh:
            cache = {r["wnid"]: r["words"] for r in csv.DictReader(fh)}
    missing = wnids - cache.keys()
    if missing:
        print(f"fetching words.txt for {len(missing)} synsets ...", file=sys.stderr)
        try:
            with urllib.request.urlopen(WORDS_URL, timeout=60) as resp:
                for line in resp.read().decode("utf-8", "replace").splitlines():
                    wnid, _, words = line.partition("\t")
                    if wnid in missing:
                        cache[wnid] = words
        except Exception as exc:  # offline is fine, the column just goes blank
            print(f"  could not fetch words.txt ({exc}); continuing", file=sys.stderr)
        WORDS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with WORDS_CACHE.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["wnid", "words"])
            w.writeheader()
            w.writerows({"wnid": k, "words": v} for k, v in sorted(cache.items()))
    return cache


def load_captions(wnids: set[str]) -> dict[str, dict]:
    if not CAPTIONS.exists():
        print(f"note: {CAPTIONS.name} missing, caption columns will be 0", file=sys.stderr)
        return {}
    with CAPTIONS.open(encoding="utf-8") as fh:
        entries = json.load(fh)
    return {Path(e["filename"]).stem: e for e in entries if e.get("wnid") in wnids}


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.0f}%" if d else "-"


def main() -> None:
    classes = discover()
    if not classes:
        sys.exit(f"no image folders under {RAW}")

    wnids = {w for _, _, w, _ in classes}
    words = synset_words(wnids)
    captions = load_captions(wnids)

    rows = []
    for pair, cls, wnid, stems in classes:
        adir = ANNOT / pair / cls
        xml = {f.stem for f in adir.rglob("*.xml")} if adir.is_dir() else set()
        matched = xml & stems

        cap = [captions[s] for s in stems if s in captions]
        tagged = [e for e in cap if e.get("tags")]
        titled = [e for e in cap if (e.get("title") or "").strip()]
        described = [e for e in cap if (e.get("description") or "").strip()]
        all_tags = [t for e in tagged for t in e["tags"]]

        tagged_stems = {s for s in stems if captions.get(s, {}).get("tags")}

        rows.append({
            "pair": pair, "class": cls, "wnid": wnid,
            "words": words.get(wnid, ""),
            "images": len(stems),
            "bbox_xml": len(xml),
            "bbox_usable": len(matched),
            "bbox_pct": pct(len(matched), len(stems)),
            "captions": len(cap),
            "caption_pct": pct(len(cap), len(stems)),
            "with_title": len(titled),
            "with_tags": len(tagged),
            "tag_pct": pct(len(tagged), len(stems)),
            "with_desc": len(described),
            "tag_tokens": len(all_tags),
            "uniq_tags": len(set(t.lower() for t in all_tags)),
            "avg_tags": f"{len(all_tags) / len(tagged):.1f}" if tagged else "0",
            # --- usable n per downstream task ---
            "task_classify": len(stems),                    # caption -> embed -> logreg
            "task_bbox_iou": len(matched),                  # predicted box vs ground truth
            "task_tag_compare": len(tagged_stems),          # SMER words vs human tags
            "task_bbox_and_tags": len(matched & tagged_stems),
        })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    cols = [
        ("pair", "pair"), ("class", "class"), ("wnid", "synset"),
        ("images", "images"), ("bbox_usable", "bbox"), ("bbox_pct", "bbox%"),
        ("captions", "captions"), ("with_tags", "tags"), ("tag_pct", "tag%"),
        ("task_bbox_and_tags", "bbox+tags"),
        ("uniq_tags", "uniq tags"), ("avg_tags", "avg tags"),
    ]
    md = ["# Data coverage", "",
          "Images **on disk**; `bbox` excludes XMLs whose image is missing. "
          "`bbox+tags` is the subset usable for both at once.", ""]
    md += ["| " + " | ".join(lbl for _, lbl in cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    md += ["| " + " | ".join(str(r[k]) for k, _ in cols) + " |" for r in rows]

    tot = {k: sum(r[k] for r in rows)
           for k in ("images", "bbox_usable", "captions", "with_tags",
                     "task_bbox_and_tags")}
    md += [f"| **TOTAL** |  |  | **{tot['images']}** | **{tot['bbox_usable']}** | "
           f"{pct(tot['bbox_usable'], tot['images'])} | **{tot['captions']}** | "
           f"**{tot['with_tags']}** | {pct(tot['with_tags'], tot['images'])} | "
           f"**{tot['task_bbox_and_tags']}** |  |  |"]

    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nwrote {OUT_CSV.relative_to(ROOT)} and {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

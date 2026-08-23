"""Reading the stage-1 caption JSONL.

The file is append-only and retries are appended rather than replacing the row
they retry, so a generation that failed once and succeeded later appears twice:

    {"stem": "n07590611_10775", "n_words": 3, "rep": 0, "caption": null,  "error": "..."}
    {"stem": "n07590611_10775", "n_words": 3, "rep": 0, "caption": "Golden crispy potatoes"}

Every stage must resolve that the same way, so it happens here once.
"""

from __future__ import annotations

import json
from pathlib import Path

Key = tuple[str, int, int]  # (stem, n_words, rep)



def load_rows(path: str | Path, keep_failed: bool = False) -> list[dict]:
    """Deduplicated rows, one per (stem, n_words, rep).

    A successful generation always beats a failed one for the same key; between
    two successes the later line wins. Rows that never succeeded are dropped
    unless keep_failed is set (useful for reporting what still needs a rerun).
    """
    best: dict[Key, dict] = {}
    for row in iter_raw(path):
        key = (row["stem"], row["n_words"], row["rep"])
        prev = best.get(key)
        if prev is None:
            best[key] = row
        elif not prev.get("caption") or row.get("caption"):
            # replace a failure with anything, or a success with a later success
            best[key] = row

    rows = list(best.values())
    if not keep_failed:
        rows = [r for r in rows if r.get("caption")]
    rows.sort(key=lambda r: (r["class"], r["stem"], r["n_words"], r["rep"]))
    return rows


def iter_raw(path: str | Path):
    """Every line as written, including duplicates -- tolerates a torn last line."""
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # a job killed mid-write can leave one partial line


def summarize(path: str | Path) -> dict:
    """Counts for a quick health check of a caption file."""
    raw = list(iter_raw(path))
    resolved = load_rows(path, keep_failed=True)
    done = [r for r in resolved if r.get("caption")]
    return {
        "lines": len(raw),
        "unique_keys": len(resolved),
        "duplicates": len(raw) - len(resolved),
        "with_caption": len(done),
        "still_failed": len(resolved) - len(done),
        "images": len({r["stem"] for r in resolved}),
        "word_counts": sorted({r["n_words"] for r in resolved}),
        "reps": sorted({r["rep"] for r in resolved}),
    }

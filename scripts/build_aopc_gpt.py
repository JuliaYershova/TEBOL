#!/usr/bin/env python3
"""Assemble the GPT-4o configuration's AOPC curves into results/metrics/aopc_gpt.csv.

The GPT-4o run predates this repository and its outputs are scattered across the
Diplom working folder. Three pairs still have their numbers in
`test_package_data*/aopc_values.csv` and those values match the published
figure. Two do not: `aopc_values.csv` for ambulance/firetruck and ant/bee comes
from a later re-run than the figure that was published, so for those two the
published curve was recovered from the stored PNGs by pixel digitisation
(axis calibrated on the gridlines; precision +/- 0.0005, about 1.5 px).

Provenance per pair is written to the `source` column. Values are the mean drop
in predicted probability after removing k words, where the words come from a
global top-20 list rather than a per-description ranking, and LIME's weights
are passed through abs(). Both differ from the Qwen3-VL pipeline.

Usage:
    python scripts/build_aopc_gpt.py
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = Path("/Users/kjzg289/Library/CloudStorage/OneDrive-AZCollaboration/"
           "study/2 semestr/Diplom/data")
OUT = ROOT / "results" / "metrics" / "aopc_gpt.csv"

#: folder -> pair, for the pairs whose CSV matches the published figure
FROM_CSV = {"test_package_data2": "acousticguitar_violin",
            "test_package_data5": "cucumber_zucchini",
            "test_package_data4": "hotpot_vase"}

#: pair -> {ranking: [value at k=1..6]}, digitised from the published PNGs
#: (test_package_data1/AOPC_ambulance_fire_truck.png and
#:  test_package_data3/AOPC_ant_bee_final1.png)
FROM_IMAGE = {
    "ambulance_firetruck": {
        "smer": [0.1550, 0.1548, 0.1552, 0.1552, 0.1552, 0.1552],
        "lime": [0.1482, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500]},
    "ant_bee": {
        "smer": [0.0909, 0.0915, 0.0915, 0.0915, 0.0915, 0.0916],
        "lime": [0.0930, 0.0925, 0.0925, 0.0925, 0.0925, 0.0925]},
}


def main() -> None:
    rows = []
    for folder, pair in FROM_CSV.items():
        with (SRC / folder / "aopc_values.csv").open() as fh:
            for r in csv.DictReader(fh):
                k = int(r["K"].strip())
                if k == 0:
                    continue
                for rank, col in (("smer", "AOPC_SMER"), ("lime", "AOPC_LIME")):
                    rows.append({"pair": pair, "arm": "caption", "setup": "w07",
                                 "ranking": rank, "k": k,
                                 "mean": float(r[col].strip()),
                                 "source": "aopc_values.csv"})
    for pair, series in FROM_IMAGE.items():
        for rank, vals in series.items():
            for k, v in enumerate(vals, start=1):
                rows.append({"pair": pair, "arm": "caption", "setup": "w07",
                             "ranking": rank, "k": k, "mean": v,
                             "source": "digitised from published PNG"})

    rows.sort(key=lambda r: (r["pair"], r["ranking"], r["k"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["pair", "arm", "setup", "ranking",
                                           "k", "mean", "source"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(rows)} rows, "
          f"{len({r['pair'] for r in rows})} pairs)")


if __name__ == "__main__":
    main()

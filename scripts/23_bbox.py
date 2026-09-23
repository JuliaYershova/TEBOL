#!/usr/bin/env python3
"""Stage 4 -- ask the captioner where the explanation's top words point.

Two modes.

`generate` reads SMER's per-word scores for one arm and setup, takes each
image's top-k words, and asks the captioner for a box:

    Provide bounding box coordinates for the object from (word, word, word).
    Format: x_min, y_min, x_max, y_max with integers only.

The words are per image, not per class -- that is the point. If SMER's top
words are the ones the classifier really used, the box lands on the object;
if they are spurious, it does not. The class name never enters the prompt.

Ranking matches 07_aopc.py's local ranking exactly: z oriented toward the
predicted class, largest first. A word appears once however many captions of
that image contain it, scored by its mean oriented z.

Only images with ImageNet ground truth are asked about, since a box with
nothing to score against costs a call and answers nothing.

Qwen3-VL answers on a 0-1000 grid and usually returns one box per word, so
every box is rescaled to pixels and all of them are kept, labelled and in
prompt order. `box` is the first -- the highest-ranked word -- and `boxes`
holds the rest for the evaluation to compare against.

`--retry-failed` re-asks only the calls whose reply could not be parsed. The
file stays append-only and the reader keeps the successful line, so a retry
never loses the original reply.

`--reps` asks the same question several times per image, as stage 1 did with
captions, so coordinate stability can be measured. Each rep gets its own seed;
temperature must be above zero or the reps are the same call five times and
measure nothing, which the script refuses to let you do by accident.

Output is append-only JSONL, one row per (image, rep), so a run killed halfway
resumes instead of starting over -- rerun the same command and it picks up
where it stopped. Progress and failures also go to a log file under logs/.
Parse failures are written as rows with box=null and the error, because the
failure rate is a result.

`draw` writes one PNG per image: the TEBOL box in red, the ImageNet box in
green, each labelled, with the IoU between them. `--all` does every image
in the pair rather than a sample. `--all-boxes` adds the other words' boxes
in orange, for seeing what the model did with the rest of the prompt.

Overlays go to artifacts/bbox/<pair>/overlays/ by default, not results/, and
for a reason: one PNG per image is ~150 KB, so a pair is well over 100 MB and
results/ is tracked in git. Pass --out to put a chosen few somewhere they can
be committed.

`table` is the results table: per class and per pair, how stable the box is
across repeated calls, how often a reply could not be localised at all, how
often the box agreed with ImageNet, and the mean IoU with a 95% interval.

The interval is computed over **images**, not over calls. Five reps of one
image are five looks at the same photograph, not five independent
observations; pooling them would divide the standard error by sqrt(5) and
report a confidence the design does not support. Each image contributes its
own mean IoU, and `summarize` -- the same helper stage 3 uses -- takes the
t-interval over those.

`sheet` puts many of them on one page, sorted by IoU, and prints three
tables: how often the asked-for box and the ImageNet box agree at the usual
thresholds, how much the box moves between reps of the same image, and which
top words localise worst. It writes results/metrics/bbox_agreement.csv with a
row per (image, rep) so the analysis can go further without rerunning the
model.

    python scripts/23_bbox.py generate --pair ant_bee --limit 5 --dry-run
    python scripts/23_bbox.py generate --pair ant_bee --limit 5
    python scripts/23_bbox.py draw --pair ant_bee --limit 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tebol import bbox as B                              # noqa: E402
from tebol.captioners import get_captioner               # noqa: E402

RAW = ROOT / "data" / "raw"
ANNOT = ROOT / "data" / "annotations"
EXPL = ROOT / "artifacts" / "explanations"
OUT = ROOT / "artifacts" / "bbox"
FIG = ROOT / "results" / "figures" / "bbox"

EMB = {"caption": "local__qwen3-embedding__4b",
       "caption_noclass": "local__qwen3-embedding__4b--noclass"}


def load_env(path: Path = ROOT / ".env") -> None:
    """Same .env handling as 01_caption.py, so both stages read one file."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def top_words(pair: str, arm: str, setup: str, k: int) -> dict[str, list[str]]:
    """stem -> its k highest-scoring words, by mean z oriented to the prediction."""
    f = EXPL / EMB[arm] / pair / arm / setup / "smer_words.parquet"
    if not f.exists():
        sys.exit(f"missing {f}")
    d = pd.read_parquet(f, columns=["stem", "cls", "word", "z", "pred_class"])

    pos = sorted(d["cls"].unique())[1]          # as in 03_train_smer.py
    d["signed"] = d["z"].where(d["pred_class"] == pos, -d["z"])

    g = (d.groupby(["stem", "word"], as_index=False)["signed"].mean()
           .sort_values(["stem", "signed"], ascending=[True, False],
                        kind="stable"))
    return {stem: list(sub["word"].head(k))
            for stem, sub in g.groupby("stem", sort=True)}


def with_truth(pair: str):
    """stem -> (image path, xml path, class), for images that have both."""
    out = {}
    for cls_dir in sorted(p for p in (RAW / pair).iterdir() if p.is_dir()):
        for img in sorted(cls_dir.glob("*.JPEG")):
            xml = ANNOT / pair / cls_dir.name / f"{img.stem}.xml"
            if xml.exists():
                out[img.stem] = (img, xml, cls_dir.name)
    return out


def done_keys(path: Path) -> set[tuple[str, int]]:
    """(stem, rep) pairs already in the output, so a rerun resumes.

    A truncated last line -- the run was killed mid-write -- is skipped rather
    than crashing the resume; that pair is simply asked again.
    """
    if not path.exists():
        return set()
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                r = json.loads(line)
                seen.add((r["stem"], r.get("rep", 0)))
            except (json.JSONDecodeError, KeyError):
                continue
    return seen


class Log:
    """Print and append to a file, so a detached run leaves a trail."""

    def __init__(self, path: Path | None):
        self.fh = None
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.fh = path.open("a", encoding="utf-8")
            self.fh.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    def __call__(self, msg: str = "") -> None:
        print(msg, flush=True)
        if self.fh:
            self.fh.write(msg + "\n")
            self.fh.flush()


def out_path(pair: str, arm: str, setup: str, model: str) -> Path:
    safe = model.replace(":", "__").replace("/", "_")
    return OUT / pair / f"{safe}__{arm}__{setup}.jsonl"


def generate(args) -> None:
    if args.reps > 1 and args.temperature == 0:
        sys.exit("--reps > 1 at temperature 0 repeats one deterministic call; "
                 "pass --temperature 1.0 (as stage 1 used) to measure stability")

    words = top_words(args.pair, args.arm, args.setup, args.top_k)
    images = with_truth(args.pair)
    stems = [s for s in sorted(images) if s in words]
    if not stems:
        sys.exit(f"{args.pair}: no image has both ground truth and SMER words")

    cap = None if args.dry_run else get_captioner(args.backend, args.model)
    model_name = "dry-run" if args.dry_run else cap.name
    path = out_path(args.pair, args.arm, args.setup, model_name)
    already = set() if args.dry_run else done_keys(path)

    n_scorable = len(stems)
    if args.limit:
        stems = stems[:args.limit]

    if args.retry_failed:
        # re-ask only the (stem, rep) pairs whose reply could not be parsed.
        # rows are appended, not rewritten; read_rows keeps the later success.
        keep = {s for s in stems}
        todo = sorted({(r["stem"], r.get("rep", 0))
                       for r in read_rows(args.pair, args.arm, args.setup,
                                          parsed_only=False)
                       if not r.get("box") and r["stem"] in keep})
    else:
        todo = [(s, r) for s in stems for r in range(args.reps)
                if (s, r) not in already]

    log = Log(None if args.dry_run else
              ROOT / "logs" / f"bbox_{args.pair}_{args.arm}_{args.setup}.log")
    limited = f", limited to {len(stems)}" if args.limit else ""
    log(f"{args.pair} {args.arm} {args.setup}: {n_scorable} scorable images"
        f"{limited} x {args.reps} rep(s), {len(todo)} calls to make")

    if args.dry_run:
        for stem, rep in todo:
            print(f"\n  {stem} rep {rep}  [{images[stem][2]}]")
            print(f"  {B.build_prompt(words[stem])}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    ok = fail = 0
    t_start = time.perf_counter()
    with path.open("a", encoding="utf-8") as fh:
        for i, (stem, rep) in enumerate(todo, 1):
            img, xml, cls = images[stem]
            feats = words[stem]
            prompt = B.build_prompt(feats)
            t0 = time.perf_counter()
            res = cap.ask(img, prompt, max_tokens=args.max_tokens,
                          temperature=args.temperature, seed=args.seed + rep)
            raw, err = B.parse_boxes(res.text)
            truths, w, h = B.truth_boxes(xml)

            boxes, over, clipped = [], False, False
            for item in raw:
                px, o = B.rescale(item["box"], w, h, args.coord_space)
                px, c = B.clip(px, w, h)
                over, clipped = over or o, clipped or c
                if px[2] > px[0] and px[3] > px[1]:
                    boxes.append({"label": item["label"], "box": px,
                                  "raw": item["box"]})
            if raw and not boxes:
                err = "every box fell outside the image after rescaling"
            box = boxes[0]["box"] if boxes else None

            row = {"stem": stem, "rep": rep, "pair": args.pair, "class": cls,
                   "arm": args.arm, "setup": args.setup, "top_k": args.top_k,
                   "features": feats, "prompt": prompt,
                   "reply": res.text, "box": box, "boxes": boxes,
                   "union": B.union([b["box"] for b in boxes]) or None,
                   "parse_error": err, "coord_space": args.coord_space,
                   "out_of_range": over, "clipped": clipped,
                   "width": w, "height": h,
                   "truth": truths, "model": cap.name,
                   "backend": args.backend, "temperature": args.temperature,
                   "seed": args.seed + rep, "error": res.error,
                   "latency_s": round(time.perf_counter() - t0, 3)}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()

            if box is None:
                fail += 1
                log(f"  [{i}/{len(todo)}] {stem} r{rep}  FAILED: "
                    f"{err or res.error}  reply={(res.text or '')[:60]!r}")
            else:
                ok += 1
                if i % args.log_every == 0 or i == len(todo):
                    rate = (time.perf_counter() - t_start) / i
                    left = (len(todo) - i) * rate
                    log(f"  [{i}/{len(todo)}] {stem} r{rep}  {box}  "
                        f"IoU {B.best_iou(box, truths):.3f}  "
                        f"{rate:.1f}s/call, ~{left/3600:.1f}h left")

    log(f"wrote {path}")
    log(f"  this run: parsed {ok}, failed {fail}")
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    good = [r for r in rows if r["box"]]
    if good:
        ious = [B.best_iou(r["box"], r["truth"]) for r in good]
        log(f"  file total: {len(rows)} rows, {len(good)} parsed, "
            f"mean IoU {sum(ious)/len(ious):.3f}, "
            f"{100*sum(i >= 0.5 for i in ious)/len(ious):.0f}% at IoU>=0.5")


def _font(px: int):
    """A readable font at `px`, falling back to PIL's bitmap one."""
    from PIL import ImageFont
    for path in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                 "/System/Library/Fonts/Helvetica.ttc",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(path, px)
        except OSError:
            continue
    return ImageFont.load_default()


def _tag(d, xy, text, fill, font, pad=3):
    """Text on a solid plate, so a label never vanishes into the photograph."""
    x, y = xy
    x0, y0, x1, y1 = d.textbbox((x, y), text, font=font)
    d.rectangle([x0 - pad, y0 - pad, x1 + pad, y1 + pad], fill=fill)
    d.text((x, y), text, fill=(255, 255, 255), font=font)
    return y1 + pad


RED, GREEN, ORANGE = (220, 20, 20), (0, 170, 0), (240, 150, 0)


def draw(args) -> None:
    from PIL import Image, ImageDraw

    rows = read_rows(args.pair, args.arm, args.setup)
    if not rows:
        sys.exit(f"no parsed boxes under {OUT / args.pair}")

    # one overlay per image: reps barely differ, so draw the chosen rep
    by_stem = {}
    for r in sorted(rows, key=lambda r: r.get("rep", 0)):
        by_stem.setdefault(r["stem"], r if r.get("rep", 0) == args.rep else None)
        if r.get("rep", 0) == args.rep:
            by_stem[r["stem"]] = r
    rows = [r for r in by_stem.values() if r]
    rows.sort(key=lambda r: r["stem"])
    if not args.all and args.limit:
        rows = rows[:args.limit]

    out_dir = Path(args.out) if args.out else OUT / args.pair / "overlays"
    out_dir.mkdir(parents=True, exist_ok=True)
    images = with_truth(args.pair)
    print(f"{args.pair}: drawing {len(rows)} image(s) -> {out_dir}")

    for i, r in enumerate(rows, 1):
        im = Image.open(images[r["stem"]][0]).convert("RGB")
        d = ImageDraw.Draw(im)
        short = min(im.size)
        wide = max(6, round(short / 55))           # bold enough to see
        fs = max(13, round(short / 26))
        font = _font(fs)
        iou = B.best_iou(r["box"], r["truth"])

        if args.all_boxes:
            for b in r.get("boxes", [])[1:]:                       # orange
                d.rectangle(b["box"], outline=ORANGE, width=max(2, wide - 1))
        for t in r["truth"]:                                       # green
            d.rectangle(t, outline=GREEN, width=wide)
        # red last, so TEBOL's box is always in front. Where the two nearly
        # coincide the green is hidden behind it -- that overlap is the
        # agreement, and the IoU in the corner is what quantifies it.
        d.rectangle(r["box"], outline=RED, width=wide)

        y = _tag(d, (6, 6), f"TEBOL ({r['features'][0]})", RED, font)
        y = _tag(d, (6, y + 5), "ImageNet ground truth", GREEN, font)
        _tag(d, (6, y + 5), f"IoU {iou:.2f}", (30, 30, 30), font)

        im.save(out_dir / f"{r['stem']}__iou{iou:.2f}.png")
        if i % 100 == 0 or i == len(rows):
            print(f"  [{i}/{len(rows)}]", flush=True)
    print(f"wrote {len(rows)} overlays to {out_dir}")


def read_rows(pair: str, arm: str, setup: str, parsed_only: bool = True):
    """Rows for one cell, one per (stem, rep).

    The file is append-only, so a retried call leaves two lines for the same
    pair. A parsed box always beats a failure and, between two parsed boxes,
    the later line wins -- the same rule captions_io.load_rows uses.
    """
    paths = sorted((OUT / pair).glob(f"*__{arm}__{setup}.jsonl"))
    best: dict[tuple[str, int], dict] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["stem"], r.get("rep", 0))
            prev = best.get(key)
            if prev is None or r.get("box") or not prev.get("box"):
                best[key] = r
    rows = list(best.values())
    return [r for r in rows if r.get("box")] if parsed_only else rows


def sheet(args) -> None:
    """One page of examples per pair, plus the agreement table."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    pairs = args.pairs or sorted(p.name for p in OUT.iterdir() if p.is_dir())
    cells = {}
    for pair in pairs:
        rows = read_rows(pair, args.arm, args.setup)
        if rows:
            cells[pair] = rows
    if not cells:
        sys.exit("nothing generated yet -- run `generate` first")

    # --- the numbers ---
    print(f"\nagreement between the asked-for box and the ImageNet box "
          f"({args.arm}, {args.setup})\n")
    print(f"  {'pair':<24}{'n':>5}{'mean IoU':>10}{'>=0.5':>8}{'>=0.75':>8}{'>=0.9':>8}")
    allious = []
    for pair, rows in cells.items():
        ious = [B.best_iou(r["box"], r["truth"]) for r in rows]
        allious += ious
        n = len(ious)
        print(f"  {pair:<24}{n:>5}{sum(ious)/n:>10.3f}"
              f"{100*sum(i>=.5 for i in ious)/n:>7.0f}%"
              f"{100*sum(i>=.75 for i in ious)/n:>7.0f}%"
              f"{100*sum(i>=.9 for i in ious)/n:>7.0f}%")
    n = len(allious)
    print(f"  {'all':<24}{n:>5}{sum(allious)/n:>10.3f}"
          f"{100*sum(i>=.5 for i in allious)/n:>7.0f}%"
          f"{100*sum(i>=.75 for i in allious)/n:>7.0f}%"
          f"{100*sum(i>=.9 for i in allious)/n:>7.0f}%")

    # --- one tidy row per (image, rep), so this need not be recomputed ---
    recs = []
    for pair, rows in cells.items():
        for r in rows:
            b = r["box"]
            recs.append({
                "pair": pair, "stem": r["stem"], "rep": r.get("rep", 0),
                "class": r["class"], "arm": r["arm"], "setup": r["setup"],
                "top_word": r["features"][0],
                "features": " ".join(r["features"]),
                "iou": B.best_iou(b, r["truth"]),
                "x0": b[0], "y0": b[1], "x1": b[2], "y1": b[3],
                "n_boxes": len(r.get("boxes", [])),
                "n_truth": len(r["truth"]),
                "width": r["width"], "height": r["height"],
            })
    df = pd.DataFrame(recs)
    MET = ROOT / "results" / "metrics"
    MET.mkdir(parents=True, exist_ok=True)
    df.to_csv(MET / "bbox_agreement.csv", index=False)
    print(f"\nwrote {MET / 'bbox_agreement.csv'}  ({len(df)} rows)")

    # --- stability: how far the box moves between reps of one image ---
    multi = df.groupby(["pair", "stem"]).filter(lambda g: len(g) > 1)
    if not multi.empty:
        g = multi.groupby(["pair", "stem"])
        st = pd.DataFrame({
            "reps": g.size(),
            "iou_sd": g["iou"].std(ddof=1),
            "iou_range": g["iou"].max() - g["iou"].min(),
            # mean corner movement as a share of image size, comparable across sizes
            "drift_pct": (g["x0"].std(ddof=1) / g["width"].first()
                          + g["x1"].std(ddof=1) / g["width"].first()
                          + g["y0"].std(ddof=1) / g["height"].first()
                          + g["y1"].std(ddof=1) / g["height"].first()) * 100 / 4,
        }).reset_index()
        print(f"\nstability across reps ({len(st)} images with >1 rep)\n")
        print(f"  {'pair':<24}{'images':>7}{'IoU sd':>9}{'IoU range':>11}{'drift %':>9}")
        for pair, sub in st.groupby("pair"):
            print(f"  {pair:<24}{len(sub):>7}{sub.iou_sd.mean():>9.3f}"
                  f"{sub.iou_range.mean():>11.3f}{sub.drift_pct.mean():>9.2f}")
    else:
        print("\nstability: only one rep per image -- rerun generate with --reps 5")

    # --- which top words localise worst ---
    w = (df.groupby("top_word")
           .agg(n=("iou", "size"), mean_iou=("iou", "mean"),
                hit=("iou", lambda v: 100 * (v >= 0.5).mean()))
           .query(f"n >= {args.min_word_n}")
           .sort_values("mean_iou"))
    if not w.empty:
        print(f"\ntop words that localise worst (>= {args.min_word_n} uses)\n")
        print(f"  {'word':<20}{'n':>5}{'mean IoU':>10}{'>=0.5':>8}")
        for word, r in w.head(args.n_words).iterrows():
            print(f"  {word:<20}{int(r.n):>5}{r.mean_iou:>10.3f}{r.hit:>7.0f}%")
        print(f"\n  best: ", end="")
        print(", ".join(f"{i} {r.mean_iou:.2f}"
                        for i, r in w.tail(5).iloc[::-1].iterrows()))

    # --- the page: worst first, so disagreement is visible rather than buried ---
    ncol = args.per_pair
    nrow = len(cells)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.5 * ncol, 2.3 * nrow),
                             squeeze=False)
    images = {p: with_truth(p) for p in cells}
    for i, (pair, rows) in enumerate(cells.items()):
        rows = sorted(rows, key=lambda r: B.best_iou(r["box"], r["truth"]),
                      reverse=not args.worst)[:ncol]
        for j in range(ncol):
            ax = axes[i][j]
            ax.set_xticks([]); ax.set_yticks([])
            if j >= len(rows):
                ax.axis("off"); continue
            r = rows[j]
            ax.imshow(Image.open(images[pair][r["stem"]][0]).convert("RGB"))
            for t in r["truth"]:                                  # green
                ax.add_patch(Rectangle((t[0], t[1]), t[2]-t[0], t[3]-t[1],
                                       fill=False, edgecolor="#00A000", lw=1.8))
            b = r["box"]                                          # red
            ax.add_patch(Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1],
                                   fill=False, edgecolor="#DC0000", lw=1.8))
            iou = B.best_iou(b, r["truth"])
            ax.set_title(f"{r['features'][0]}  IoU {iou:.2f}", fontsize=8,
                         color="#111" if iou >= 0.5 else "#B00")
            if j == 0:
                ax.set_ylabel(pair.replace("_", " /\n"), fontsize=8)
    which = "worst" if args.worst else "best"
    fig.suptitle(f"Asked-for box (red) against ImageNet ground truth (green) "
                 f"\u2014 {which} {ncol} per pair, {args.arm} {args.setup}",
                 fontsize=11, y=1.0)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    stem = f"contact_sheet_{args.arm}_{args.setup}_{which}"
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    print(f"\nwrote {FIG / (stem + '.png')}")
    plt.close(fig)


def table(args) -> None:
    """Four tables: IoU, success rate, stability, and words never localised."""
    from tebol.stats import summarize

    pairs = args.pairs or sorted(p.name for p in OUT.iterdir() if p.is_dir())
    recs, wrecs = [], []
    for pair in pairs:
        for r in read_rows(pair, args.arm, args.setup, parsed_only=False):
            b = r.get("box")
            iou = B.best_iou(b, r["truth"]) if b else float("nan")
            recs.append({
                "pair": pair, "class": r["class"], "stem": r["stem"],
                "rep": r.get("rep", 0), "top_word": r["features"][0],
                "parsed": bool(b), "iou": iou,
                "hit": bool(b) and iou >= args.thr,
                "x0": b[0] if b else float("nan"),
                "y0": b[1] if b else float("nan"),
                "x1": b[2] if b else float("nan"),
                "y1": b[3] if b else float("nan"),
                "width": r["width"], "height": r["height"],
            })
            # per-word attribution needs the model to have labelled its boxes
            labels = {x["label"] for x in r.get("boxes", []) if x["label"]}
            # re-derived from the stored reply, so existing runs need no redo
            refused = B.refused_labels(r.get("reply"))
            for i, word in enumerate(r["features"]):
                w = word.lower()
                wrecs.append({
                    "pair": pair, "class": r["class"], "word": w,
                    "position": i + 1, "reply_parsed": bool(b),
                    "labelled": bool(labels or refused),
                    "got_box": w in labels,
                    "refused": w in refused,
                    "absent": not (w in labels or w in refused),
                })
    if not recs:
        sys.exit("nothing generated yet -- run `generate` first")
    df, wdf = pd.DataFrame(recs), pd.DataFrame(wrecs)
    MET = ROOT / "results" / "metrics"
    MET.mkdir(parents=True, exist_ok=True)

    def rows_for(keys):
        out = []
        for key, g in (df.groupby(keys) if keys else [("all", df)]):
            img = g.groupby("stem")
            st = summarize(img["iou"].mean().to_numpy())
            st["ci_lo"], st["ci_hi"] = max(0., st["ci_lo"]), min(1., st["ci_hi"])
            sd = img["iou"].std(ddof=1)
            drift = ((img["x0"].std(ddof=1) + img["x1"].std(ddof=1))
                     / img["width"].first()
                     + (img["y0"].std(ddof=1) + img["y1"].std(ddof=1))
                     / img["height"].first()) * 100 / 4
            agree = img["hit"].apply(lambda v: v.nunique() == 1)
            reps = img.size()
            out.append({
                "key": " / ".join(key) if isinstance(key, tuple) else str(key),
                "images": img.ngroups, "calls": len(g),
                "reps": reps.mean(),
                "iou_mean": st["mean"], "iou_sd": st["sd"],
                "ci_lo": st["ci_lo"], "ci_hi": st["ci_hi"],
                "half": (st["ci_hi"] - st["ci_lo"]) / 2,
                "hits": int(g.hit.sum()), "misses": int((~g.hit).sum()),
                "hit_pct": 100 * g.hit.mean(),
                "no_coords": int((~g.parsed).sum()),
                "no_coords_pct": 100 * (~g.parsed).mean(),
                "iou_sd_within": sd.mean() if (reps > 1).any() else float("nan"),
                "drift_pct": drift.mean() if (reps > 1).any() else float("nan"),
                "agree_pct": (100 * agree.mean() if (reps > 1).any()
                              else float("nan")),
            })
        return pd.DataFrame(out)

    by_class = rows_for(["pair", "class"])
    by_pair = rows_for(["pair"])
    overall = rows_for(None)
    summary = pd.concat([by_class, by_pair, overall], keys=["class", "pair", "all"],
                        names=["level"]).reset_index(level=0)
    summary.to_csv(MET / "bbox_summary.csv", index=False)

    n = lambda v, f: "      -" if pd.isna(v) else format(v, f)
    W = max(len(r.key) for t in (by_class, by_pair, overall)
            for r in t.itertuples()) + 2

    print(f"\nbounding boxes against ImageNet -- {args.arm} {args.setup}")
    print(f"  success = IoU >= {args.thr} | interval over images, not calls | "
          f"IoU sd, drift and agree are within-image across reps\n")
    print("  " + " " * W
          + f"{'images':>7}{'calls':>7}"
          + f"{'mean IoU':>10}{'+-':>7}{'95% CI':>18}"
          + f"{'success':>9}"
          + f"{'IoU sd':>8}{'drift%':>8}{'agree':>7}")
    for t, gap in ((by_class, True), (by_pair, True), (overall, False)):
        for r in t.itertuples():
            print(f"  {r.key:<{W}}"
                  f"{r.images:>7}{r.calls:>7}"
                  f"{r.iou_mean:>10.3f}{r.half:>7.3f}"
                  f"   [{r.ci_lo:.3f}, {r.ci_hi:.3f}]"
                  f"{r.hit_pct:>8.1f}%"
                  f"{n(r.iou_sd_within, '>8.3f')}{n(r.drift_pct, '>8.2f')}"
                  f"{n(r.agree_pct, '>6.0f')}%")
        if gap:
            print()

    # --- 4. top-3 words the model never gave coordinates for ---
    lab = wdf[wdf.labelled]
    share = 100 * len(lab) / len(wdf)
    w = (lab.groupby(["class", "word"])
            .agg(in_top3=("got_box", "size"),
                 boxed=("got_box", "sum"),
                 refused=("refused", "sum"),
                 absent=("absent", "sum"))
            .assign(refused_pct=lambda d: 100 * d.refused / d.in_top3,
                    no_box_pct=lambda d: 100 * (d.refused + d.absent) / d.in_top3)
            .query(f"in_top3 >= {args.min_word_n} and (refused + absent) > 0")
            .sort_values("no_box_pct", ascending=False))
    w.to_csv(MET / "bbox_words_no_coords.csv")
    print(f"\ntop-3 words the model gave no coordinates for\n")
    print(f"  Attribution needs the reply to name its boxes, which {share:.0f}% do;"
          f"\n  the rest give bare numbers. This is that {share:.0f}% subset, "
          f"words seen >= {args.min_word_n} times.\n"
          f"  refused = the model answered 0,0,0,0 or -1,-1,-1,-1 -- an explicit"
          f"\n            'not in this picture'."
          f"\n  absent  = no box and no denial, usually because another word "
          f"already\n            covered the same object.\n")
    if w.empty:
        print("  none -- every top-3 word got its own box")
    else:
        rows = list(w.head(args.n_words).iterrows())
        cw = max(len(c) for (c, _), _ in rows) + 2
        print(f"  {'class':<{cw}}{'word':<16}{'in top 3':>10}{'boxed':>8}"
              f"{'refused':>9}{'absent':>8}{'no box':>9}")
        for (cls, word), r in rows:
            print(f"  {cls:<{cw}}{word:<16}{int(r.in_top3):>10}{int(r.boxed):>8}"
                  f"{int(r.refused):>9}{int(r.absent):>8}{r.no_box_pct:>8.1f}%")

    print(f"\nwrote {MET / 'bbox_summary.csv'}"
          f" and {MET / 'bbox_words_no_coords.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    common = dict(pair=dict(required=True), )
    for name in ("generate", "draw", "sheet", "table"):
        s = sub.add_parser(name)
        s.add_argument("--pair", required=name in ("generate", "draw"))
        s.add_argument("--arm", default="caption", choices=sorted(EMB))
        s.add_argument("--setup", default="w05")
        s.add_argument("--limit", type=int, default=None)

    g = sub.choices["generate"]
    g.add_argument("--top-k", type=int, default=3)
    g.add_argument("--backend", default="local", choices=["local", "openai"])
    g.add_argument("--model", default=None)
    g.add_argument("--reps", type=int, default=1,
                   help="calls per image; >1 needs --temperature above 0")
    g.add_argument("--temperature", type=float, default=0.0)
    g.add_argument("--seed", type=int, default=0,
                   help="base seed; rep r uses seed+r")
    g.add_argument("--log-every", type=int, default=25,
                   help="log one progress line every N successful calls")
    g.add_argument("--max-tokens", type=int, default=512,
                   help="128 truncated 0.8%% of replies before the coordinates, "
                        "usually when a stop word made the model argue first")
    g.add_argument("--retry-failed", action="store_true",
                   help="re-ask only the calls whose reply could not be parsed")
    g.add_argument("--coord-space", type=int, default=B.COORD_SPACE,
                   help="grid the model answers on; 1000 for Qwen3-VL, "
                        "pass the image size for a model that replies in pixels")
    g.add_argument("--dry-run", action="store_true",
                   help="print the prompts without calling the model")

    d = sub.choices["draw"]
    d.add_argument("--model", default=None)
    d.add_argument("--truth", action="store_true", default=True,
                   help="also draw the ground-truth box in green")
    d.add_argument("--no-truth", dest="truth", action="store_false")
    d.add_argument("--all-boxes", action="store_true",
                   help="also draw the other words' boxes in orange")
    d.add_argument("--all", action="store_true",
                   help="every image in the pair, not just --limit")
    d.add_argument("--rep", type=int, default=0,
                   help="which repetition to draw")
    d.add_argument("--out", default=None,
                   help="output directory (default artifacts/bbox/<pair>/overlays)")

    sh = sub.choices["sheet"]
    sh.add_argument("--pairs", nargs="*", default=None)
    sh.add_argument("--per-pair", type=int, default=5)
    sh.add_argument("--worst", action="store_true",
                    help="show the worst matches instead of the best")
    sh.add_argument("--min-word-n", type=int, default=5,
                    help="ignore top words used fewer than this many times")
    sh.add_argument("--n-words", type=int, default=12)

    tb = sub.choices["table"]
    tb.add_argument("--pairs", nargs="*", default=None)
    tb.add_argument("--thr", type=float, default=0.7,
                    help="IoU at or above which a box counts as localised")
    tb.add_argument("--min-word-n", type=int, default=20)
    tb.add_argument("--n-words", type=int, default=15)

    args = ap.parse_args()
    load_env()
    {"generate": generate, "draw": draw, "sheet": sheet,
     "table": table}[args.mode](args)


if __name__ == "__main__":
    main()

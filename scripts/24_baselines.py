#!/usr/bin/env python3
"""Stage 5 -- baselines the SMER pipeline has to beat.

The stage-3 numbers say a logistic regression over caption embeddings
separates the pairs. They do not say whether the captions were needed. This
runs the comparisons that answer that, one `--baseline` at a time.

    zeroshot     ask Qwen3-VL, in words, which of the two classes it sees
    clip         CLIP ViT-B/32, nearest class prompt by cosine similarity
    coca         CoCa ViT-B/32, the same, from a contrastive captioner
    clip_probe   the same CLIP image vectors, with the stage-3 classifier on top
    coca_probe   the same for CoCa
    strmatch     read the class name out of the caption, no model at all

**zeroshot** offers both class names and asks the model to pick one.

    What is on the image: ant or bee? Answer with just the class name.

Offering the names is what makes this comparable, not what makes it easy: the
classifier is a binary logistic regression trained on those two classes and
never has to consider a wasp either. Both sit against the same 50% floor. The
asymmetry that remains is supervision -- the classifier needed ~2,600 labelled
images, this needs none -- so a zero-shot score matching the classifier says
the labels bought little.

**clip** and **coca** answer the same question without generating anything.
Both were trained to put an image and its caption near each other in one
vector space, so classifying is a nearest-neighbour lookup: embed the image,
embed "a photo of an ant." and "a photo of a bee.", take whichever is closer.
No training, no labels, 150M parameters against Qwen3-VL's 32B.

The template matters and is not decoration. CLIP was trained on web captions,
which are sentences, so "a photo of a {name}." scores better than the bare
word; `--template "{}"` runs the bare version if you want to show the gap
rather than assert it. There is no answer order to randomise here -- cosine
similarity is symmetric -- which removes a confound the generative baseline
has to control for.

**clip_probe** and **coca_probe** are the reviewers' baseline, and the only
one that is structurally identical to the pipeline. The encoder is frozen and
never sees a class name: it turns each image into 512 numbers, exactly as
Qwen3-Embedding turns each caption into 2560. Those vectors then go to the
same LogisticRegression, on the same 25 folds, with the same labels and the
same image-level scoring.

So everything is held fixed except what was encoded -- a description of the
photograph, or the photograph. That isolates what routing through language
costs or buys, which zero-shot cannot: zero-shot hands CLIP the class names
and trains nothing, so it differs from the pipeline in two ways at once.

**strmatch** is the floor the pipeline has to clear. Nothing is predicted:
the classifier is thrown away and the caption text is searched for either
class name. If exactly one is there, that is the answer. No embeddings, no
training, a dozen lines of string matching.

The matcher is imported from scripts/class_leakage.py, so it is the same
token-level matching that produced the ablated captions -- plurals, hyphen
parts and multiword phrases handled, never substrings, so "ant" does not fire
on "plant". `--max-tier` chooses how much counts as naming the class: 1 the
class name and its inflections, 2 adding agreed synonyms, 3 adding the
standalone head noun.

Captions that name both classes, or neither, are not predictions. They are
counted as wrong, which is the strict reading, and the abstention rate is
reported beside the accuracy because it is the more interesting number: it is
the share of images whose caption carries no label to read. At image level the
five repetitions vote, matching how the classifier averages their
probabilities.

Names come from data/manifest/synset_words.csv -- ImageNet's own word for the
synset, not a label chosen here. For zeroshot, which name is offered first is
decided by a hash of the stem, so a model that favours the first option cannot
favour one class; `first_offered` is recorded either way, so the bias can be
tested for rather than assumed away.

**Scoring is on the stage-3 folds.** A zero-shot model has no training set, so
it could be scored once over all images -- but then its number and the
classifier's would not be comparable, one being a single proportion and the
other a mean over 25 fits. Instead the same (repeat, fold) partitions from
results/folds/ are applied to the predictions, giving 25 test-set accuracies
that line up with the classifier's. The spread is smaller by construction: a
zero-shot prediction does not change between repeats, so only the partition
varies, and the interval is narrower than the classifier's for that reason
rather than because the baseline is more certain.

Output is append-only JSONL, one row per image, resumable.

    python scripts/24_baselines.py run --pair ant_bee --limit 5 --dry-run
    python scripts/24_baselines.py run --pair ant_bee
    python scripts/24_baselines.py table
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tebol.captioners import get_captioner          # noqa: E402
from tebol.stats import summarize                   # noqa: E402

RAW = ROOT / "data" / "raw"
FOLDS = ROOT / "results" / "folds"
OUT = ROOT / "artifacts" / "baselines"
MET = ROOT / "results" / "metrics"
SYNSETS = ROOT / "data" / "manifest" / "synset_words.csv"
TERMS = ROOT / "data" / "manifest" / "class_terms.csv"

PROMPT = "What is on the image: {a} or {b}? Answer with just the class name."

#: open_clip architecture and pretrained tag for the similarity baselines.
#: Weights are loaded from artifacts/openclip_weights/ if present -- the
#: folder holds checkpoints for any open_clip architecture, CLIP and CoCa
#: alike -- because this
#: network's TLS proxy presents a CA certificate OpenSSL 3 rejects, so
#: huggingface_hub cannot download while curl can. See README of that folder.
CLIP_MODELS = {
    "clip": ("ViT-B-32", "openai", "ViT-B-32.pt"),
    "coca": ("coca_ViT-B-32", "laion2b_s13b_b90k", "coca_ViT-B-32.bin"),
}
WEIGHTS = ROOT / "artifacts" / "openclip_weights"
PROBES = {"clip_probe": "clip", "coca_probe": "coca"}
EMB = ROOT / "artifacts" / "probe_embeddings"
TEMPLATE = "a photo of {article} {}."


def load_env(path: Path = ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def class_names() -> dict[str, str]:
    """class -> the name shown to the model.

    ImageNet's own words for the synset, but among its synonyms the one this
    project also recognises. n03345487 is "fire engine, fire truck": its first
    word is "fire engine", which class_terms.csv does not list, so a model
    answering exactly what it was offered scored as naming neither class and
    put that pair at chance. Picking the overlap keeps the name the prompt
    shows and the name the reply is read against the same string.
    """
    wnid, known = {}, {}
    for r in csv.DictReader(TERMS.open()):
        wnid[r["class"]] = r["wnid"]
        if int(r["tier"]) == 1:
            known.setdefault(r["class"], set()).add(r["term"].strip().lower())
    words = {r["wnid"]: [w.strip() for w in r["words"].split(",")]
             for r in csv.DictReader(SYNSETS.open())}
    out = {}
    for cls, w in wnid.items():
        if w not in words:
            continue
        cand = words[w]
        out[cls] = next((c for c in cand if c.lower() in known.get(cls, set())),
                        cand[0])
    return out


def accept_terms() -> dict[str, set[str]]:
    """class -> the strings that count as naming it, for reading the reply.

    Tier-1 terms from class_terms.csv, plus the name the prompt actually
    offered. Those two lists disagree: the prompt uses ImageNet's canonical
    word, and for n03345487 that is "fire engine", while class_terms lists
    "firetruck" and "fire truck". Without this the model answering "fire
    engine" -- the exact string it was offered -- scored as naming neither
    class, which put that pair at chance.
    """
    out: dict[str, set[str]] = {}
    for r in csv.DictReader(TERMS.open()):
        if int(r["tier"]) == 1:
            out.setdefault(r["class"], set()).add(r["term"].strip().lower())
    for cls, shown in class_names().items():
        out.setdefault(cls, set()).add(shown.lower())
    return out


def images_of(pair: str):
    """(stem, path, class) for every image of a pair, deterministic order."""
    for d in sorted(p for p in (RAW / pair).iterdir() if p.is_dir()):
        for img in sorted(d.glob("*.JPEG")):
            yield img.stem, img, d.name


def read_pred(text: str | None, a: str, b: str, terms) -> tuple[str | None, str]:
    """(predicted class, note). Naming both or neither is not a prediction."""
    if not text:
        return None, "empty reply"
    t = " " + text.lower().replace(",", " ").replace(".", " ") + " "
    hits = [c for c in (a, b)
            if any(f" {w} " in t for w in terms.get(c, {c}))]
    if len(hits) == 1:
        return hits[0], ""
    return None, ("named both classes" if hits else "named neither class")


def out_path(pair: str, baseline: str, model: str) -> Path:
    return OUT / pair / f"{baseline}__{model.replace(':', '__')}.jsonl"


def done_stems(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.add(json.loads(line)["stem"])
            except (json.JSONDecodeError, KeyError):
                continue
    return out


def run(args) -> None:
    names, terms = class_names(), accept_terms()
    items = list(images_of(args.pair))
    classes = sorted({c for _, _, c in items})
    if len(classes) != 2:
        sys.exit(f"{args.pair}: expected 2 classes, found {classes}")

    cap = None if args.dry_run else get_captioner(args.backend, args.model)
    model_name = "dry-run" if args.dry_run else cap.name
    path = out_path(args.pair, args.baseline, model_name)
    already = set() if args.dry_run else done_stems(path)

    todo = [it for it in items if it[0] not in already]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{args.pair} {args.baseline}: {len(items)} images, "
          f"{len(already)} done, {len(todo)} to ask about", flush=True)

    if args.dry_run:
        for stem, _, cls in todo[:args.limit or 5]:
            print(f"\n  {stem}  [{cls}]\n  {build(args.baseline, stem, classes, names)[0]}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    t0, ok, bad = time.perf_counter(), 0, 0
    with path.open("a", encoding="utf-8") as fh:
        for i, (stem, img, cls) in enumerate(todo, 1):
            prompt, a = build(args.baseline, stem, classes, names)
            res = cap.ask(img, prompt, max_tokens=args.max_tokens,
                          temperature=args.temperature, seed=args.seed)
            pred, note = read_pred(res.text, *classes, terms)
            fh.write(json.dumps({
                "stem": stem, "pair": args.pair, "class": cls,
                "baseline": args.baseline, "first_offered": a,
                "named_a_class": pred is not None,
                "prompt": prompt, "reply": res.text, "pred": pred,
                "correct": (pred == cls) if pred else None, "note": note,
                "model": cap.name, "temperature": args.temperature,
                "seed": args.seed, "error": res.error,
            }, ensure_ascii=False) + "\n")
            fh.flush()
            ok += pred is not None
            bad += pred is None
            if i % args.log_every == 0 or i == len(todo):
                rate = (time.perf_counter() - t0) / i
                print(f"  [{i}/{len(todo)}] {stem} -> {pred}  "
                      f"{rate:.1f}s/call, ~{(len(todo)-i)*rate/3600:.1f}h left",
                      flush=True)
    print(f"\nwrote {path}\n  answered {ok}, unreadable {bad}")


def build(baseline: str, stem: str, classes, names) -> tuple[str, str]:
    """(prompt, name offered first), ordered by a hash so position cannot bias."""
    flip = int(hashlib.md5(stem.encode()).hexdigest(), 16) & 1
    a, b = (classes[1], classes[0]) if flip else (classes[0], classes[1])
    return PROMPT.format(a=names[a], b=names[b]), names[a]


def fill(template: str, name: str) -> str:
    """Template with the class name, choosing a/an so the prompt is grammatical.

    Not cosmetic: CLIP was trained on written captions, and "a photo of a ant."
    is not English. The article is part of the text it has to match.
    """
    art = "an" if name[0].lower() in "aeiou" else "a"
    return template.replace("{article}", art).format(name)


def load_clip(arch: str, tag: str, local: Path, dev: str):
    """(model, preprocess), from a local weight file when one is there.

    OpenAI's ViT-B-32.pt is a TorchScript archive rather than a state dict, so
    it needs open_clip's own converter; every other checkpoint is a plain
    state dict. Falling back to the tag keeps this working on a machine whose
    network lets huggingface_hub through.
    """
    import torch
    import open_clip
    from open_clip.openai import build_model_from_openai_state_dict
    from open_clip.transform import image_transform

    if not local.exists():
        m, _, pp = open_clip.create_model_and_transforms(
            arch, pretrained=tag, device=dev)
        return m, pp

    if local.suffix == ".pt":                       # TorchScript, OpenAI's format
        sd = torch.jit.load(local, map_location="cpu").state_dict()
        # OpenAI shipped these weights in fp16; MPS will not mix dtypes, and
        # fp32 costs nothing at this size
        m = build_model_from_openai_state_dict(sd).float()
        size = m.visual.image_size
        pp = image_transform(size, is_train=False,
                             mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711))
        return m, pp

    m, _, pp = open_clip.create_model_and_transforms(arch, pretrained=str(local))
    return m, pp


def run_clip(args) -> None:
    """CLIP or CoCa: embed once, classify by cosine similarity. Local, batched."""
    import torch
    import open_clip
    from PIL import Image

    arch, tag, fname = CLIP_MODELS[args.baseline]
    dev = ("mps" if torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available() else "cpu")
    local = WEIGHTS / fname
    print(f"loading {arch} ({tag}) on {dev}", flush=True)
    model, preprocess = load_clip(arch, tag, local, dev)
    model.eval().to(dev)
    tok = open_clip.get_tokenizer(arch)

    names = class_names()
    items = list(images_of(args.pair))
    classes = sorted({c for _, _, c in items})
    if args.limit:
        items = items[:args.limit]

    prompts = [fill(args.template, names[c]) for c in classes]
    print(f"{args.pair}: {len(items)} images | prompts {prompts}", flush=True)
    with torch.no_grad():
        txt = model.encode_text(tok(prompts).to(dev)).float()
        txt /= txt.norm(dim=-1, keepdim=True)

    path = out_path(args.pair, args.baseline, f"{arch}__{tag}")
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with path.open("w", encoding="utf-8") as fh:      # local and fast: no resume
        for i in range(0, len(items), args.batch):
            chunk = items[i:i + args.batch]
            batch = torch.stack([preprocess(Image.open(f).convert("RGB"))
                                 for _, f, _ in chunk]).to(dev)
            with torch.no_grad():
                img = model.encode_image(batch).float()
                img /= img.norm(dim=-1, keepdim=True)
                sim = (img @ txt.T).cpu().numpy()
            for (stem, _, cls), row in zip(chunk, sim):
                pred = classes[int(row.argmax())]
                fh.write(json.dumps({
                    "stem": stem, "pair": args.pair, "class": cls,
                    "baseline": args.baseline, "first_offered": None,
                    "prompt": " | ".join(prompts), "reply": None,
                    "pred": pred, "correct": pred == cls, "note": "",
                    "sim": {c: round(float(v), 5) for c, v in zip(classes, row)},
                    "margin": round(float(abs(row[0] - row[1])), 5),
                    "model": f"{arch}:{tag}", "temperature": None,
                    "template": args.template,
                    "seed": None, "error": None,
                }, ensure_ascii=False) + "\n")
            done = min(i + args.batch, len(items))
            if done % (args.batch * 10) == 0 or done == len(items):
                r = (time.perf_counter() - t0) / done
                print(f"  [{done}/{len(items)}] {r*1000:.0f}ms/image, "
                      f"~{(len(items)-done)*r/60:.1f}min left", flush=True)

    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    acc = sum(r["correct"] for r in rows) / len(rows)
    print(f"\nwrote {path}\n  raw accuracy over all {len(rows)} images: {acc:.4f}")


def run_strmatch(args) -> None:
    """Predict from the class words in the caption. No model, no training."""
    import importlib.util
    from collections import Counter
    from tebol.captions_io import load_rows

    spec = importlib.util.spec_from_file_location(
        "class_leakage", ROOT / "scripts" / "class_leakage.py")
    cl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cl)
    single, phrase, _ = cl.load_terms(args.max_tier)

    files = sorted(f for f in (ROOT / "artifacts" / "captions" / args.pair)
                   .glob("*.jsonl") if not f.stem.endswith(cl.NOCLASS))
    if not files:
        sys.exit(f"no captions for {args.pair}")
    rows = [r for r in load_rows(files[0]) if r["n_words"] == args.words]
    if not rows:
        sys.exit(f"{args.pair}: no captions at {args.words} words")
    classes = sorted({r["class"] for r in rows})

    path = out_path(args.pair, f"{args.baseline}_t{args.max_tier}",
                    f"w{args.words:02d}")
    path.parent.mkdir(parents=True, exist_ok=True)
    per_image: dict[str, list] = {}
    n_both = n_none = 0
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            toks = r.get("tokens") or []
            # which classes does this caption name? a decision needs exactly one
            hits = [c for c in classes if cl.find(toks, c, single, phrase)]
            pred = hits[0] if len(hits) == 1 else None
            n_both += len(hits) > 1
            n_none += not hits
            per_image.setdefault(r["stem"], []).append((pred, r["class"]))
            fh.write(json.dumps({
                "stem": r["stem"], "rep": r["rep"], "pair": args.pair,
                "class": r["class"], "baseline": args.baseline,
                "words": args.words, "max_tier": args.max_tier,
                "caption": r.get("caption_norm") or r.get("caption"),
                "named": hits, "pred": pred, "correct": pred == r["class"],
                "note": "" if pred else ("both" if hits else "neither"),
                "model": None, "first_offered": None, "prompt": None,
                "reply": None, "error": None,
            }, ensure_ascii=False) + "\n")

    # image level: the five repetitions vote, abstentions do not get a say
    img_rows = []
    for stem, v in per_image.items():
        truth = v[0][1]
        votes = Counter(p for p, _ in v if p)
        top = votes.most_common()
        pred = top[0][0] if len(top) == 1 or (
            len(top) > 1 and top[0][1] > top[1][1]) else None
        img_rows.append((pred, truth))
    acc = np.mean([p == t for p, t in img_rows])
    absent = np.mean([p is None for p, t in img_rows])
    print(f"{args.pair} strmatch tier<={args.max_tier} w{args.words:02d}: "
          f"{len(rows)} captions, {len(img_rows)} images")
    print(f"  captions naming both classes {n_both} ({100*n_both/len(rows):.1f}%), "
          f"neither {n_none} ({100*n_none/len(rows):.1f}%)")
    print(f"  image-level accuracy {acc:.4f}, abstained on "
          f"{100*absent:.1f}% of images (counted wrong)")
    print(f"wrote {path}")


def run_probe(args) -> None:
    """Frozen image vectors, then the stage-3 classifier on the stage-3 folds."""
    import torch
    import open_clip
    from PIL import Image
    from sklearn.linear_model import LogisticRegression

    base = PROBES[args.baseline]
    arch, tag, fname = CLIP_MODELS[base]
    dev = ("mps" if torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available() else "cpu")

    items = list(images_of(args.pair))
    if args.limit:
        items = items[:args.limit]
    stems = [s for s, _, _ in items]
    y = np.array([c for _, _, c in items])

    # --- embed once, cache: the encoder is frozen, so this never changes ---
    EMB.mkdir(parents=True, exist_ok=True)
    cache = EMB / f"{base}__{args.pair}.npz"
    if cache.exists() and not args.refresh:
        z = np.load(cache, allow_pickle=True)
        X, cached = z["X"], list(z["stems"])
        if cached != stems:
            sys.exit(f"{cache} was built for a different image set; --refresh")
        print(f"loaded {X.shape} from {cache}")
    else:
        print(f"embedding {len(items)} images with {arch} on {dev}", flush=True)
        model, preprocess = load_clip(arch, tag, WEIGHTS / fname, dev)
        model.eval().to(dev)
        out, t0 = [], time.perf_counter()
        for i in range(0, len(items), args.batch):
            chunk = items[i:i + args.batch]
            b = torch.stack([preprocess(Image.open(f).convert("RGB"))
                             for _, f, _ in chunk]).to(dev)
            with torch.no_grad():
                out.append(model.encode_image(b).float().cpu().numpy())
            done = min(i + args.batch, len(items))
            if done % (args.batch * 20) == 0 or done == len(items):
                print(f"  [{done}/{len(items)}]", flush=True)
        X = np.concatenate(out)
        np.savez_compressed(cache, X=X, stems=np.array(stems))
        print(f"  {time.perf_counter()-t0:.0f}s, wrote {cache}")

    # --- the stage-3 classifier, on the stage-3 folds ---
    folds = pd.read_csv(FOLDS / f"{args.pair}.csv")
    pos = sorted(set(y))[1]
    idx = {s: i for i, s in enumerate(stems)}
    path = out_path(args.pair, args.baseline, f"{arch}__{tag}")
    path.parent.mkdir(parents=True, exist_ok=True)
    accs = []
    with path.open("w", encoding="utf-8") as fh:
        for (rep, fold), g in folds.groupby(["repeat", "fold"]):
            te = np.array([idx[s] for s in g.stem if s in idx])
            tr = np.setdiff1d(np.array([idx[s] for s in folds[
                folds.repeat == rep].stem if s in idx]), te)
            clf = LogisticRegression(max_iter=2000,
                                     class_weight="balanced").fit(X[tr], y[tr])
            pred = clf.predict(X[te])
            accs.append(float((pred == y[te]).mean()))
            for j, pr in zip(te, pred):
                fh.write(json.dumps({
                    "stem": stems[j], "pair": args.pair, "class": str(y[j]),
                    "baseline": args.baseline, "repeat": int(rep),
                    "fold": int(fold), "pred": str(pr),
                    "correct": bool(pr == y[j]),
                    "model": f"{arch}:{tag}", "first_offered": None,
                    "prompt": None, "reply": None, "note": "", "error": None,
                }, ensure_ascii=False) + "\n")
    st = summarize(np.array(accs))
    print(f"\nwrote {path}\n  {len(accs)} folds, accuracy {st['mean']:.4f} "
          f"[{st['ci_lo']:.4f}, {st['ci_hi']:.4f}]")


def table(args) -> None:
    """Every baseline against every arm, paired on the same 25 folds.

    Both caption arms are shown, because neither alone is the comparison.
    The classifier learns the two classes from training labels; CLIP and CoCa
    are given them in the prompt. What the full caption arm has on top is the
    captioner having already written the class word into the text, which is
    what `caption_noclass` removes. Read together they bracket the answer:
    the full arm is the pipeline as it runs, the ablated arm is the pipeline
    with the captioner's own recognition taken out of the input.

    Differences are paired by (repeat, fold) and tested with a t-test on the
    25 paired differences, so a lead of half a point is not read as a win when
    the folds disagree about it.
    """
    from scipy import stats

    pairs = args.pairs or sorted(p.name for p in OUT.iterdir() if p.is_dir())
    arms = ["caption", "caption_noclass"]
    lr = pd.read_csv(MET / "folds.csv")
    lr = lr[(lr.level == "image") & lr.arm.isin(arms) & (lr.setup == args.setup)]

    rows = []
    for base in args.baselines:
        for pair in pairs:
            glob = (f"strmatch_t{args.strmatch_tier}__*.jsonl"
                    if base == "strmatch" else f"{base}__*.jsonl")
            files = sorted((OUT / pair).glob(glob))
            if not files:
                continue
            rows_ = [json.loads(l) for f in files
                     for l in f.read_text(encoding="utf-8").splitlines()
                     if l.strip()]
            if base == "strmatch":
                # per caption, so the five repetitions vote to one prediction
                # per image, as the classifier averages their probabilities.
                # A tie or an all-abstain image has no prediction and is wrong.
                from collections import Counter
                by = {}
                for r in rows_:
                    by.setdefault(r["stem"], []).append(r)
                rows_ = []
                for stem, v in by.items():
                    c = Counter(r["pred"] for r in v if r["pred"])
                    top = c.most_common()
                    pred = (top[0][0] if top and
                            (len(top) == 1 or top[0][1] > top[1][1]) else None)
                    rows_.append({"stem": stem, "class": v[0]["class"],
                                  "correct": pred == v[0]["class"]})
            fm = pd.read_csv(FOLDS / f"{pair}.csv")

            if base in PROBES:
                # A probe is refit on every fold, so its 25 accuracies differ
                # for the same reason the classifier's do: a different training
                # subset. Interval over those folds, as for the classifier.
                d = pd.DataFrame(rows_)
                b = {k: float(g.correct.mean())
                     for k, g in d.groupby(["repeat", "fold"])}
                st = summarize(np.array(list(b.values())))
                lo, hi, acc = st["ci_lo"], st["ci_hi"], st["mean"]
                n_img = d.stem.nunique()
            else:
                # A zero-shot prediction never changes, so the folds only
                # re-cut fixed answers into fifths -- their spread is 2-3x
                # narrower than the real sampling error. The uncertainty is
                # over which images were collected, so bootstrap those. The
                # folds are still used for the paired differences below, where
                # they are what makes the two methods comparable.
                pred = {r["stem"]: bool(r["correct"]) for r in rows_}
                b = {k: np.mean([pred[s] for s in sub.stem if s in pred])
                     for k, sub in fm.groupby(["repeat", "fold"])
                     if any(s in pred for s in sub.stem)}
                v = np.array([pred[s] for s in fm.stem.unique() if s in pred])
                rng = np.random.default_rng(args.boot_seed)
                draws = v[rng.integers(0, v.size,
                                       (args.boot, v.size))].mean(axis=1)
                lo, hi = np.percentile(draws, [2.5, 97.5])
                acc, n_img = float(v.mean()), int(v.size)

            rec = {"baseline": base, "pair": pair, "setup": args.setup,
                   "images": n_img, "base_acc": acc,
                   "base_lo": float(lo), "base_hi": float(hi),
                   "interval": "folds" if base in PROBES else "bootstrap"}
            for arm in arms:
                g = lr[(lr.pair == pair) & (lr.arm == arm)] \
                    .set_index(["repeat", "fold"])["accuracy"]
                keys = [k for k in b if k in g.index]
                l = np.array([g.loc[k] for k in keys])
                d = l - np.array([b[k] for k in keys])
                se = d.std(ddof=1) / np.sqrt(d.size)
                h = se * stats.t.ppf(0.975, d.size - 1)
                _, pv = stats.ttest_rel(l, np.array([b[k] for k in keys]))
                rec |= {f"{arm}_acc": l.mean(), f"{arm}_diff": d.mean(),
                        f"{arm}_lo": d.mean() - h, f"{arm}_hi": d.mean() + h,
                        f"{arm}_p": pv}
            rows.append(rec)
    if not rows:
        sys.exit("nothing generated yet -- run `run` first")
    t = pd.DataFrame(rows)
    f = MET / f"baselines_{args.setup}.csv"
    t.to_csv(f, index=False)

    W = max(len(r.pair) for r in t.itertuples()) + 2
    # the method, its own ablation, then the baselines from cheapest to
    # most capable: no model, zero-shot, then the same classifier on frozen
    # image vectors. Fixed order so the table reads the same every run.
    ORDER = ["caption", "caption_noclass", "strmatch", "zeroshot",
             "clip", "clip_probe", "coca", "coca_probe"]
    have = set(args.baselines) | {"caption", "caption_noclass"}
    cols = [c for c in ORDER if c in have]
    head = {"caption": "LR caption", "caption_noclass": "LR no-class",
            "clip_probe": "CLIP probe", "coca_probe": "CoCa probe",
            "strmatch": "name in text"}

    # one row per pair, one column per method, each with its 95% half-width
    lr_ci = {}
    for pair in pairs:
        for arm in arms:
            v = lr[(lr.pair == pair) & (lr.arm == arm)]["accuracy"].to_numpy()
            if v.size:
                st = summarize(v)
                lr_ci[(pair, arm)] = (st["mean"], (st["ci_hi"] - st["ci_lo"]) / 2)

    print(f"\naccuracy at {args.setup}, image level, same images\n")
    print("  " + " " * (W + 7)
          + "".join(f"{head.get(c, c.upper()):>19}" for c in cols))
    for pair in pairs:
        sub = t[t.pair == pair]
        if sub.empty:
            continue
        line = f"  {pair:<{W}}{int(sub.iloc[0].images):>7}"
        for c in cols:
            if c in arms:
                m, h = lr_ci.get((pair, c), (float("nan"),) * 2)
            else:
                r = sub[sub.baseline == c]
                m, h = ((r.iloc[0].base_acc,
                         (r.iloc[0].base_hi - r.iloc[0].base_lo) / 2)
                        if not r.empty else (float("nan"),) * 2)
            line += "                  -" if pd.isna(m) else f"{m:>13.4f} ±{h:.3f}"
        print(line)
    best = max(cols, key=lambda c: 0)     # column order is the reading order
    legend = [
        ("name in text", "string matching, no model: the class whose name "
                         "the caption contains,", "class name and inflections "
                         "only, the five repetitions voting."),

        ("CLIP / COCA", "zero-shot: the two class names go in the prompt,",
                        "nothing is trained."),
        ("*_PROBE", "the same frozen image vectors, never shown a class name,",
                    "with the stage-3 LogisticRegression on top."),
    ]
    legend = [
        ("LR caption", "the method: the classifier on the mean word embedding "
                       "of each", "caption, class word and all."),
        ("LR no-class", "the same, on captions with the class word deleted.",
                        "isolates what the description alone carries."),
    ] + legend
    print()
    for name, first, second in legend:
        print(f"  {name:<16}{first}")
        print(f"  {'':<16}{second}")
    print(f"\n  Every trained column uses the same 25 folds, the same labels "
          f"and the same\n  image-level scoring. Only the vector differs.\n"
          f"\n  Intervals differ in kind and are not comparable as widths: a "
          f"t-interval\n  over 25 folds where a model is refit, "
          f"{args.boot} bootstrap resamples over\n  images where it is not.\n"
          f"\n  paired differences and significance: {f}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("run", "table"):
        s = sub.add_parser(name)
        s.add_argument("--baseline", default="zeroshot",
                       choices=["zeroshot", "clip", "coca",
                                "clip_probe", "coca_probe",
                                "strmatch"])

    r = sub.choices["run"]
    r.add_argument("--pair", required=True)
    r.add_argument("--backend", default="local", choices=["local", "openai"])
    r.add_argument("--model", default=None)
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--max-tokens", type=int, default=32)
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--log-every", type=int, default=25)
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--batch", type=int, default=64, help="clip/coca only")
    r.add_argument("--max-tier", type=int, default=1, choices=[1, 2, 3],
                   help="strmatch: 1 class name, 2 +synonyms, 3 +head noun")
    r.add_argument("--words", type=int, default=5,
                   help="strmatch: caption length to read")
    r.add_argument("--refresh", action="store_true",
                   help="re-embed instead of using the cached vectors")
    r.add_argument("--template", default=TEMPLATE,
                   help="clip/coca prompt template; \"{}\" for the bare name")

    t = sub.choices["table"]
    t.add_argument("--pairs", nargs="*", default=None)
    t.add_argument("--baselines", nargs="*", default=["clip", "coca", "zeroshot"])
    t.add_argument("--boot", type=int, default=5000,
                   help="bootstrap resamples for the baseline interval")
    t.add_argument("--boot-seed", type=int, default=0)
    t.add_argument("--strmatch-tier", type=int, default=1, choices=[1, 2, 3],
                   help="which strmatch run to show; 1 is the class name only")
    t.add_argument("--setup", default="w05",
                   help="which caption length to compare against")

    args = ap.parse_args()
    load_env()
    if args.mode == "table":
        table(args)
    elif args.baseline == "strmatch":
        run_strmatch(args)
    elif args.baseline in PROBES:
        run_probe(args)
    elif args.baseline in CLIP_MODELS:
        run_clip(args)
    else:
        run(args)


if __name__ == "__main__":
    main()

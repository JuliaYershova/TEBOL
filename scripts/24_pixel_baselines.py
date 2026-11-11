#!/usr/bin/env python3
"""Stage 4b -- pixel-space attribution baselines, scored against the same boxes.

Section 6.7 scores TEBOL's localisation against ImageNet's annotations. On its
own that number has nothing to sit against, so the two attribution methods the
earlier work illustrated are run here on the same images and scored the same
way:

    gradcam_resnet   Grad-CAM over ResNet50's last convolutional block
    lime_resnet      LIME over superpixels, same ResNet50 classifier
    lime_vgg16       LIME over superpixels, VGG16 classifier

Each produces a saliency map rather than a box, so a box has to be derived, and
the derivation needs a free parameter in both cases. Rather than pick one and
invite the question of whether it was tuned, the model is run once per image
and boxes are derived at every setting of that parameter:

    Grad-CAM   mask = cam >= tau * cam.max(), largest connected component,
               its bounding box; tau swept over 0.10 .. 0.50
    LIME       the k highest-weighted positive superpixels, bounding box of
               their union; k swept over 1 .. 8

Both are also scored by the pointing game, which needs no threshold at all:
whether the single most salient point falls inside an annotated box. For
Grad-CAM that is the arg max of the map, for LIME the centroid of the
highest-weighted positive superpixel.

**The target class is the ground-truth class, not the prediction.** This gives
the baselines their best case: a heatmap for a class the model got wrong would
measure classification, not localisation. The backbone's own top-1 agreement
with the annotation is recorded per image so the two can be separated.

**Images are resized to 224x224 without cropping.** The usual Resize(256) plus
CenterCrop(224) discards a border, and an annotated object can lie in it, so
the crop would charge the method for pixels it was never shown. Aspect ratio
is distorted instead, and box coordinates map back linearly.

Output is append-only JSONL under artifacts/pixel_baselines/, one row per
(image, method), so a run killed halfway resumes rather than restarting.

    python scripts/24_pixel_baselines.py run --method gradcam_resnet
    python scripts/24_pixel_baselines.py run --method lime_resnet --n 200
    python scripts/24_pixel_baselines.py evaluate
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# LIME draws a tqdm bar per image; with thousands of images that is megabytes
# of carriage returns in the log and nothing else.
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tebol import bbox as B  # noqa: E402

RAW = ROOT / "data" / "raw"
ANNOT = ROOT / "data" / "annotations"
OUT_DIR = ROOT / "artifacts" / "pixel_baselines"
MET = ROOT / "results" / "metrics"

IMG_EXT = {".jpeg", ".jpg", ".png"}
SIZE = 224
SEED = 20261015

#: Our ten classes as ImageNet-1k indices, matched through the synset words in
#: data/manifest/synset_words.csv against the torchvision category list.
CLASS_INDEX = {
    "acousticguitar": 402,   # acoustic guitar      n02676566
    "violin": 889,           # violin, fiddle       n04536866
    "ambulance": 407,        # ambulance            n02701002
    "firetruck": 555,        # fire engine          n03345487
    "ant": 310,              # ant                  n02219486
    "bee": 309,              # bee                  n02206856
    "cucumber": 943,         # cucumber, cuke       n07718472
    "zucchini": 939,         # zucchini, courgette  n07716358
    "hotpot": 926,           # hot pot, hotpot      n07590611
    "vase": 883,             # vase                 n04522168
}

#: the baseline's own colour, kept clear of the red TEBOL uses and of
#: the green annotation; dark enough to read over a jet heatmap
BLUE = (20, 50, 190)

TAUS = [round(0.10 + 0.05 * i, 2) for i in range(9)]   # 0.10 .. 0.50
KS = list(range(1, 9))                                  # 1 .. 8
METHODS = ("gradcam_resnet", "lime_resnet", "lime_vgg16")


def images() -> list[tuple[str, str, str, Path, Path]]:
    """(pair, class, stem, image path, xml path) for annotated images."""
    out = []
    for pair_dir in sorted(p for p in RAW.iterdir() if p.is_dir()):
        for cls_dir in sorted(c for c in pair_dir.iterdir() if c.is_dir()):
            for img in sorted(cls_dir.iterdir()):
                if img.suffix.lower() not in IMG_EXT:
                    continue
                xml = ANNOT / pair_dir.name / cls_dir.name / f"{img.stem}.xml"
                if xml.exists():
                    out.append((pair_dir.name, cls_dir.name, img.stem, img, xml))
    return out


def sample(rows, n_per_class: int, seed: int = SEED):
    """A seeded, class-stratified subset, so every method sees one image set."""
    if not n_per_class:
        return rows
    rng = np.random.default_rng(seed)
    out = []
    by_class: dict[str, list] = {}
    for r in rows:
        by_class.setdefault(r[1], []).append(r)
    for cls in sorted(by_class):
        g = sorted(by_class[cls], key=lambda r: r[2])
        idx = rng.permutation(len(g))[:n_per_class]
        out.extend(g[i] for i in sorted(idx))
    return out


# --------------------------------------------------------------------------
# models


def load_model(name: str, device):
    import torchvision.models as tvm
    if name == "resnet":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        layer = m.layer4[-1]
    elif name == "vgg16":
        m = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1)
        layer = m.features[-1]
    else:
        raise ValueError(name)
    return m.eval().to(device), layer


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_tensor(arr: np.ndarray, device):
    """[N,H,W,3] uint8 or float in 0-1 -> normalised float tensor [N,3,H,W]."""
    import torch
    x = arr.astype(np.float32)
    if x.max() > 1.5:
        x /= 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(x.transpose(0, 3, 1, 2)).to(device)


def load_image(path: Path) -> tuple[np.ndarray, int, int]:
    """224x224 RGB uint8, plus the original width and height."""
    from PIL import Image
    im = Image.open(path).convert("RGB")
    w, h = im.size
    return np.asarray(im.resize((SIZE, SIZE), Image.BILINEAR)), w, h


# --------------------------------------------------------------------------
# saliency -> box


def box_from_mask(mask: np.ndarray, w: int, h: int, largest: bool = True):
    """Bounding box of a saliency mask, in original-image pixels.

    `largest` keeps only the biggest connected component, which is what the
    weakly-supervised localisation literature does with a thresholded CAM: the
    map is one blob plus noise, and the noise should not stretch the box.
    LIME is the opposite case -- its top-k superpixels are a deliberate set and
    may well be disconnected, so taking one component would mean the box never
    grew with k. There it is the union that is wanted.
    """
    from skimage.measure import label, regionprops
    if not mask.any():
        return None
    if largest:
        region = max(regionprops(label(mask)), key=lambda r: r.area)
        y0, x0, y1, x1 = region.bbox      # [min_row, min_col, max_row, max_col)
    else:
        ys, xs = np.nonzero(mask)
        y0, x0, y1, x1 = ys.min(), xs.min(), ys.max() + 1, xs.max() + 1
    sx, sy = w / SIZE, h / SIZE
    return [int(x0 * sx), int(y0 * sy), int(round(x1 * sx)), int(round(y1 * sy))]



def saliency_scores(s: np.ndarray, truths, w: int, h: int) -> dict:
    """Threshold-free agreement between a saliency map and the annotation.

    IoU scores a box we derived ourselves, so it measures our thresholding as
    much as the method. These measure the map directly. `s` is at 224x224 and
    non-negative; truth boxes are in original pixels and are rasterised onto
    the same grid.

        containment  share of the saliency mass lying inside the annotation.
                     A uniform map scores |T|/A, so that is the null.
        lift         containment divided by that null: 1.0 is uninformative,
                     higher means the mass is concentrated on the object.
        coverage     share of the annotation receiving above-uniform saliency,
                     which separates "covers the whole object" from "found one
                     corner of it".
        auc          pixel-level ROC-AUC of the map predicting membership of
                     the annotation; 0.5 is chance and no threshold is chosen.

    A box is the special case of a map that is uniform inside it, so TEBOL is
    scored by the same code: containment is then precision and coverage recall.
    """
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    sx, sy = SIZE / w, SIZE / h
    for t in truths:
        x0 = max(0, min(SIZE - 1, int(t[0] * sx)))
        y0 = max(0, min(SIZE - 1, int(t[1] * sy)))
        x1 = max(x0 + 1, min(SIZE, int(round(t[2] * sx))))
        y1 = max(y0 + 1, min(SIZE, int(round(t[3] * sy))))
        mask[y0:y1, x0:x1] = True
    total = float(s.sum())
    area = mask.mean()
    if total <= 0 or area <= 0:
        return {"containment": float("nan"), "lift": float("nan"),
                "coverage": float("nan"), "auc": float("nan"),
                "truth_area": area}
    p = s / total
    containment = float(p[mask].sum())
    coverage = float((p[mask] > 1.0 / p.size).mean())
    flat, lab = s.ravel(), mask.ravel()
    order = np.argsort(flat, kind="stable")
    ranks = np.empty(len(flat), dtype=np.float64)
    ranks[order] = np.arange(1, len(flat) + 1)
    # average ranks over ties, so a flat map scores exactly 0.5
    _, inv, counts = np.unique(flat, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    n_pos, n_neg = int(lab.sum()), int((~lab).sum())
    auc = ((ranks[lab].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
           if n_pos and n_neg else float("nan"))
    return {"containment": containment, "lift": containment / area,
            "coverage": coverage, "auc": float(auc), "truth_area": float(area)}


def gradcam(model, layer, img: np.ndarray, target: int, device) -> np.ndarray:
    """Grad-CAM map at 224x224, normalised to 0-1."""
    import torch
    import torch.nn.functional as F
    acts, grads = {}, {}

    def fwd(_m, _i, o):
        acts["v"] = o.detach()

    def bwd(_m, _gi, go):
        grads["v"] = go[0].detach()

    h1 = layer.register_forward_hook(fwd)
    h2 = layer.register_full_backward_hook(bwd)
    try:
        x = to_tensor(img[None], device).requires_grad_(True)
        logits = model(x)
        model.zero_grad(set_to_none=True)
        logits[0, target].backward()
        a, g = acts["v"][0], grads["v"][0]                    # [C,h,w]
        cam = F.relu((g.mean(dim=(1, 2))[:, None, None] * a).sum(0))
        cam = F.interpolate(cam[None, None], size=(SIZE, SIZE),
                            mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-12)
        pred = int(logits[0].argmax())
        return cam.cpu().numpy(), pred
    finally:
        h1.remove(); h2.remove()


def lime_explain(model, img: np.ndarray, target: int, device,
                 num_samples: int, batch: int = 64):
    """LIME over SLIC superpixels. Returns (segments, weights, pred)."""
    import torch
    from lime.lime_image import LimeImageExplainer
    from skimage.segmentation import slic

    def predict(arrs: np.ndarray) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in range(0, len(arrs), batch):
                x = to_tensor(arrs[i:i + batch], device)
                out.append(torch.softmax(model(x), dim=1).cpu().numpy())
        return np.concatenate(out)

    seg_fn = lambda a: slic(a, n_segments=50, compactness=10.0, start_label=0)
    expl = LimeImageExplainer(verbose=False)
    e = expl.explain_instance(img, predict, labels=(target,), top_labels=None,
                              hide_color=0, num_samples=num_samples,
                              segmentation_fn=seg_fn,
                              batch_size=batch, random_seed=SEED)
    weights = dict(e.local_exp[target])
    pred = int(np.argmax(predict(img[None])[0]))
    return e.segments, weights, pred


# --------------------------------------------------------------------------
# run


def run(method: str, rows, num_samples: int, resume: bool) -> None:
    import torch
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if method == "tebol":
        model = layer = None
    else:
        backbone = "vgg16" if method.endswith("vgg16") else "resnet"
        model, layer = load_model(backbone, device)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{method}.jsonl"
    seen = set()
    if resume and path.exists():
        with path.open() as fh:
            for line in fh:
                try:
                    seen.add(json.loads(line)["stem"])
                except Exception:
                    pass
        print(f"resuming: {len(seen)} already done")

    todo = [r for r in rows if r[2] not in seen]
    print(f"{method}: {len(todo)} images to do on {device}")
    t0 = time.time()
    with path.open("a") as fh:
        for i, (pair, cls, stem, img_path, xml) in enumerate(todo, 1):
            try:
                img, w, h = load_image(img_path)
                truths, _, _ = B.truth_boxes(xml)
                target = CLASS_INDEX[cls]
                rec = {"method": method, "pair": pair, "class": cls,
                       "stem": stem, "width": w, "height": h, "truth": truths,
                       "target": target}
                if method.startswith("gradcam"):
                    cam, pred = gradcam(model, layer, img, target, device)
                    py, px = np.unravel_index(int(cam.argmax()), cam.shape)
                    rec["point"] = [int(px * w / SIZE), int(py * h / SIZE)]
                    rec["boxes"] = {str(t): box_from_mask(cam >= t * cam.max(), w, h)
                                    for t in TAUS}
                    rec.update(saliency_scores(cam, truths, w, h))
                else:
                    segs, weights, pred = lime_explain(
                        model, img, target, device, num_samples)
                    order = [s for s, v in sorted(weights.items(),
                                                  key=lambda kv: -kv[1]) if v > 0]
                    boxes = {}
                    for k in KS:
                        mask = np.isin(segs, order[:k]) if order else np.zeros_like(segs, bool)
                        boxes[str(k)] = box_from_mask(mask, w, h, largest=False)
                    rec["boxes"] = boxes
                    rec["n_positive"] = len(order)
                    sal = np.zeros(segs.shape, dtype=np.float64)
                    for sid, wt in weights.items():
                        if wt > 0:
                            sal[segs == sid] = wt
                    rec.update(saliency_scores(sal, truths, w, h))
                    if order:
                        ys, xs = np.nonzero(segs == order[0])
                        rec["point"] = [int(xs.mean() * w / SIZE),
                                        int(ys.mean() * h / SIZE)]
                    else:
                        rec["point"] = None
                rec["pred"] = pred
                rec["pred_correct"] = bool(pred == target)
                fh.write(json.dumps(rec) + "\n")
            except Exception as exc:                     # noqa: BLE001
                fh.write(json.dumps({"method": method, "pair": pair,
                                     "class": cls, "stem": stem,
                                     "error": str(exc)}) + "\n")
            fh.flush()
            if i % 25 == 0:
                rate = (time.time() - t0) / i
                print(f"  {i}/{len(todo)}  {rate:.2f}s/img  "
                      f"eta {rate * (len(todo) - i) / 60:.0f} min", flush=True)
    print(f"wrote {path}")


# --------------------------------------------------------------------------
# evaluate


def evaluate(restrict_common: bool) -> None:
    import pandas as pd
    frames = {}
    for m in METHODS:
        p = OUT_DIR / f"{m}.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(l) for l in p.open()]
        frames[m] = [r for r in rows if "error" not in r]
        print(f"{m}: {len(frames[m])} rows ({len(rows) - len(frames[m])} failed)")
    if not frames:
        sys.exit("nothing to evaluate; run first")

    common = set()
    if restrict_common and len(frames) > 1:
        common = set.intersection(*({r["stem"] for r in v} for v in frames.values()))
        frames = {m: [r for r in v if r["stem"] in common] for m, v in frames.items()}
        print(f"restricted to {len(common)} images common to every method")

    # TEBOL is scored from its stored boxes, on the same images
    teb = [r for r in tebol_rows()
           if not restrict_common or r["stem"] in common] if BBOX_ART.exists() else []
    if teb:
        frames["tebol"] = teb
        print(f"tebol: {len(teb)} rows from stored boxes")

    sweep, best = [], []
    for m, rows in frames.items():
        if m == "tebol":
            ious = np.array([r["iou"] for r in rows])
            sweep.append({"method": m, "param": float("nan"), "n": len(ious),
                          "iou_mean": ious.mean(),
                          "hit_pct": 100 * (ious >= 0.5).mean()})
        else:
            params = TAUS if m.startswith("gradcam") else KS
            for p in params:
                ious = []
                for r in rows:
                    b = r["boxes"].get(str(p))
                    ious.append(B.best_iou(b, r["truth"]) if b else 0.0)
                ious = np.array(ious)
                sweep.append({"method": m, "param": p, "n": len(ious),
                              "iou_mean": ious.mean(),
                              "hit_pct": 100 * (ious >= 0.5).mean()})
        pt = [r for r in rows if r.get("point")]
        point_acc = 100 * np.mean([
            any(b[0] <= r["point"][0] <= b[2] and b[1] <= r["point"][1] <= b[3]
                for b in r["truth"]) for r in pt]) if pt else float("nan")
        s = pd.DataFrame([x for x in sweep if x["method"] == m])
        top = s.loc[s.iou_mean.idxmax()]
        def avg(key):
            v = [r[key] for r in rows if isinstance(r.get(key), (int, float))
                 and not np.isnan(r[key])]
            return float(np.mean(v)) if v else float("nan")

        best.append({"method": m, "n": len(rows),
                     "best_param": top["param"], "iou_mean": top["iou_mean"],
                     "hit_pct": top["hit_pct"], "point_pct": point_acc,
                     "containment": avg("containment"), "lift": avg("lift"),
                     "coverage": avg("coverage"), "auc": avg("auc"),
                     "truth_area": avg("truth_area"),
                     "top1_pct": 100 * np.mean(
                         [r["pred_correct"] for r in rows if "pred_correct" in r])
                     if any("pred_correct" in r for r in rows) else float("nan")})

    MET.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sweep).to_csv(MET / "pixel_baselines_sweep.csv", index=False)
    bf = pd.DataFrame(best)
    bf.to_csv(MET / "pixel_baselines.csv", index=False)
    print("\nbest setting per method (parameter chosen by mean IoU):")
    print(bf.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nwrote {MET / 'pixel_baselines.csv'} and _sweep.csv")



# --------------------------------------------------------------------------
# draw -- side-by-side examples for the paper


BBOX_ART = ROOT / "artifacts" / "bbox"
FIG_DIR = ROOT / "results" / "figures" / "pixel_baselines"


def tebol_box(pair: str, stem: str):
    """TEBOL's box for one image: the first repetition, which is rep 0."""
    for p in sorted((BBOX_ART / pair).glob("*caption__w05.jsonl")):
        with p.open() as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("stem") == stem and r.get("box") and r.get("rep") == 0:
                    return r["box"], r.get("features"), r.get("truth")
    return None, None, None



def tebol_rows():
    """Score TEBOL's boxes with the same measures as the saliency maps.

    A box is a map that is uniform inside it and zero outside, so no model has
    to be re-run: containment becomes the box's precision against the
    annotation and coverage its recall. Every repetition is scored, matching
    how Section 6.7 reports the mean over five.
    """
    out = []
    for pair_dir in sorted(BBOX_ART.glob("*")):
        for p in sorted(pair_dir.glob("*caption__w05.jsonl")):
            for line in p.open():
                r = json.loads(line)
                b, truths = r.get("box"), r.get("truth")
                if not b or not truths:
                    continue
                w, h = r["width"], r["height"]
                sal = np.zeros((SIZE, SIZE))
                sx, sy = SIZE / w, SIZE / h
                x0 = max(0, min(SIZE - 1, int(b[0] * sx)))
                y0 = max(0, min(SIZE - 1, int(b[1] * sy)))
                x1 = max(x0 + 1, min(SIZE, int(round(b[2] * sx))))
                y1 = max(y0 + 1, min(SIZE, int(round(b[3] * sy))))
                sal[y0:y1, x0:x1] = 1.0
                rec = {"method": "tebol", "pair": r["pair"], "class": r["class"],
                       "stem": r["stem"], "rep": r.get("rep", 0),
                       "iou": B.best_iou(b, truths),
                       "point": [(b[0] + b[2]) // 2, (b[1] + b[3]) // 2],
                       "truth": truths}
                rec.update(saliency_scores(sal, truths, w, h))
                out.append(rec)
    return out


def draw(stems: list[str], tau: float, k: int, out_name: str,
         methods: list[str] | None = None,
         method_box: bool = True) -> None:
    """One row per image, one column per method, ground truth in green.

    Saliency is recomputed here rather than read back, because the JSONL keeps
    boxes and not maps -- a handful of images is seconds of work and storing
    every heatmap would not be.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from skimage.segmentation import mark_boundaries

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    index = {r[2]: r for r in images()}
    rows = [index[s] for s in stems if s in index]
    if not rows:
        sys.exit("none of those stems have annotations")

    want = methods or ["tebol", "gradcam", "lime_resnet", "lime_vgg16"]
    need = {"gradcam": ["resnet"], "lime_resnet": ["resnet"],
            "lime_vgg16": ["vgg16"], "tebol": []}
    models = {b: load_model(b, device)
              for b in {x for m in want for x in need[m]}}

    titles = {"tebol": "TEBOL", "gradcam": "Grad-CAM (ResNet50)",
              "lime_resnet": "LIME (ResNet50)", "lime_vgg16": "LIME (VGG16)"}
    cols = [titles[m] for m in want]
    fig, axes = plt.subplots(len(rows), len(cols),
                             figsize=(3.1 * len(cols), 3.2 * len(rows)),
                             squeeze=False)

    for ri, (pair, cls, stem, img_path, xml) in enumerate(rows):
        img, w, h = load_image(img_path)
        truths, _, _ = B.truth_boxes(xml)
        target = CLASS_INDEX[cls]
        panels = []

        for m in want:
            if m == "tebol":
                box, feats, _ = tebol_box(pair, stem)
                sal = None
                if box:
                    sal = np.zeros((SIZE, SIZE))
                    sal[int(box[1] * SIZE / h):max(1, int(box[3] * SIZE / h)),
                        int(box[0] * SIZE / w):max(1, int(box[2] * SIZE / w))] = 1
                panels.append((img, None, box,
                               ", ".join(feats[:3]) if feats else "", sal))
            elif m == "gradcam":
                cam, _ = gradcam(*models["resnet"], img, target, device)
                panels.append((img, cam,
                               box_from_mask(cam >= tau * cam.max(), w, h),
                               f"tau = {tau}", cam))
            else:
                backbone = "resnet" if m.endswith("resnet") else "vgg16"
                segs, weights, _ = lime_explain(models[backbone][0], img,
                                                target, device, 1000)
                order = [s for s, v in sorted(weights.items(),
                                              key=lambda kv: -kv[1]) if v > 0]
                mask = (np.isin(segs, order[:k]) if order
                        else np.zeros_like(segs, bool))
                sal = np.zeros(segs.shape)
                for sid, wt in weights.items():
                    if wt > 0:
                        sal[segs == sid] = wt
                panels.append((mark_boundaries(img / 255.0, mask.astype(int),
                                               color=(1, 1, 0)),
                               None, box_from_mask(mask, w, h, largest=False),
                               f"k = {k}", sal))

        for ci, (base, cam_map, bx, note, sal) in enumerate(panels):
            ax = axes[ri][ci]
            ax.imshow(base if base.ndim == 3 else base, extent=[0, w, h, 0])
            if cam_map is not None:
                ax.imshow(cam_map, extent=[0, w, h, 0], cmap="jet", alpha=0.45)
            handles = []
            for t in truths:
                ax.add_patch(Rectangle((t[0], t[1]), t[2] - t[0], t[3] - t[1],
                                       fill=False, edgecolor="#00C000", lw=2.2))
            handles.append(Line2D([], [], color="#00C000", lw=2.2,
                                  label="ImageNet annotation"))
            if bx and method_box:
                ax.add_patch(Rectangle((bx[0], bx[1]), bx[2] - bx[0],
                                       bx[3] - bx[1], fill=False,
                                       edgecolor="#D00000", lw=2.0, ls="--"))
                handles.append(Line2D([], [], color="#D00000", lw=2.0,
                                      ls="--", label=f"{cols[ci]} box"))
            leg = ax.legend(handles=handles, loc="upper right", fontsize=7.5,
                            framealpha=0.75, handlelength=1.6,
                            borderpad=0.35, labelspacing=0.3)
            leg.get_frame().set_edgecolor("none")
            if sal is not None:
                sc = saliency_scores(sal, truths, w, h)
                txt = f"containment {sc['containment']:.2f}"
                if bx and method_box:
                    txt = f"IoU {B.best_iou(bx, truths):.2f}   " + txt
                ax.text(0.03, 0.03, txt, transform=ax.transAxes, fontsize=9,
                        color="white",
                        bbox=dict(facecolor="#000000AA", pad=2, lw=0))
            if ri == 0:
                ax.set_title(cols[ci], fontsize=11)
            if ci == 0:
                ax.set_ylabel(f"{cls}\n{stem}", fontsize=8)
            ax.text(0.97, 0.03, note, transform=ax.transAxes, fontsize=7.5,
                    color="white", ha="right",
                    bbox=dict(facecolor="#000000AA", pad=2, lw=0))
            ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{out_name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG_DIR / out_name}.pdf  "
          f"(green = ImageNet annotation, dashed red = method)")




def _tebol_rows_rep0():
    """Every image's first repetition, with its IoU attached."""
    ann = {r[2] for r in images()}
    out = []
    for pair_dir in sorted(BBOX_ART.glob("*")):
        for f in sorted(pair_dir.glob("*caption__w05.jsonl")):
            for line in f.open():
                r = json.loads(line)
                if (r.get("box") and r.get("truth") and r.get("rep") == 0
                        and r["stem"] in ann):
                    r["iou"] = B.best_iou(r["box"], r["truth"])
                    out.append(r)
    return out


def _plural(word: str) -> bool:
    return word.endswith("s") and not word.endswith("ss")


def pick_examples(mode: str, per_class: int = 1, max_area: float = 0.75):
    """Example images per class, selected once so every method shows the same.

    `accurate` deliberately avoids a perfect score. At IoU 1.00 the two boxes
    coincide and the figure shows one rectangle, which demonstrates nothing to
    a reader asked to compare them; the window below keeps them distinct while
    still being plainly a success.

    `random` draws one image per class with a fixed seed and no condition on
    the score, so the figure shows what the method typically does rather than
    what it does at its best or worst. The other two modes are selected on
    IoU and are not representative; this one is.

    `inaccurate` excludes the multi-instance case -- a plural ranked word, or
    an image with several annotated objects -- because there the attribution
    named the class correctly and only the choice of instance differs. What
    remains are failures of the ranking itself, where the highest-ranked word
    describes an attribute or the setting rather than the object.

    Images whose annotation fills most of the frame are excluded throughout: a
    box covering everything is correct for uninteresting reasons.
    """
    rows = {}
    for r in _tebol_rows_rep0():
        w, h = r["width"], r["height"]
        area = sum((t[2] - t[0]) * (t[3] - t[1]) for t in r["truth"]) / (w * h)
        if area > max_area:
            continue
        if mode == "accurate":
            if not 0.70 <= r["iou"] <= 0.96:
                continue
            key = -r["iou"]
        elif mode == "inaccurate":
            if (r["iou"] > 0.30 or len(r["truth"]) > 1
                    or _plural(r["features"][0])):
                continue
            key = r["iou"]
        else:
            key = 0.0
        rows.setdefault(r["class"], []).append((key, r["stem"]))
    out = []
    for cls in sorted(rows):
        if mode == "random":
            rng = np.random.default_rng(SEED)
            stems = sorted(st for _, st in rows[cls])
            idx = rng.permutation(len(stems))[:per_class]
            out += [stems[i] for i in sorted(idx)]
        else:
            out += [st for _, st in sorted(rows[cls])[:per_class]]
    return out


#: The paper sets its body text in Linux Libertine, which is not installed
#: here; STIX Two Text is the nearest available face and is what the figures
#: use so their labels sit beside the prose rather than against it.
SERIF = ("/System/Library/Fonts/Supplemental/STIXTwoText.ttf",
         "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
         "/System/Library/Fonts/Supplemental/Georgia.ttf")


def _serif(px: int):
    from PIL import ImageFont
    for path in SERIF:
        try:
            return ImageFont.truetype(path, px)
        except OSError:
            continue
    return ImageFont.load_default()


def overlay(stems: list[str], tau: float, k: int, method: str,
            out_dir: Path | None = None, method_box: bool = False) -> None:
    """Single-image overlays in the style of 23_bbox.py's `draw`.

    Same plates, same colours, same geometry, so a baseline figure can sit
    beside a TEBOL one without the reader having to re-learn the legend:
    green is always the ImageNet annotation, red always the method's own box,
    and the plate in the corner names what produced it. Grad-CAM additionally
    paints its heatmap under the boxes.
    """
    import importlib.util
    import torch
    from PIL import Image

    spec = importlib.util.spec_from_file_location(
        "bbox_draw", ROOT / "scripts" / "23_bbox.py")
    bx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bx)          # reuse _tag, _font, RED, GREEN

    from PIL import ImageDraw
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    backbone = "vgg16" if method.endswith("vgg16") else "resnet"
    model, layer = load_model(backbone, device)

    index = {r[2]: r for r in images()}
    out_dir = out_dir or (FIG_DIR / method)
    out_dir.mkdir(parents=True, exist_ok=True)

    label = {"gradcam_resnet": "Grad-CAM (ResNet50)",
             "lime_resnet": "LIME (ResNet50)",
             "lime_vgg16": "LIME (VGG16)",
             "tebol": None}[method]

    for stem in stems:
        if stem not in index:
            print(f"  {stem}: no annotation, skipped")
            continue
        pair, cls, _, img_path, xml = index[stem]
        arr, w, h = load_image(img_path)
        truths, _, _ = B.truth_boxes(xml)
        target = CLASS_INDEX[cls]
        im = Image.open(img_path).convert("RGB")

        if method == "tebol":
            box, sal, note = None, None, ""
        elif method.startswith("gradcam"):
            cam, _ = gradcam(model, layer, arr, target, device)
            box = box_from_mask(cam >= tau * cam.max(), w, h)
            sal, note = cam, f"tau {tau}"
            # paint the heatmap under the boxes, at the image's own size
            import matplotlib.cm as cm
            heat = cm.jet(np.asarray(Image.fromarray(
                (cam * 255).astype(np.uint8)).resize((w, h),
                Image.BILINEAR)) / 255.0)[..., :3]
            blend = (0.55 * np.asarray(im) / 255.0 + 0.45 * heat).clip(0, 1)
            im = Image.fromarray((blend * 255).astype(np.uint8))
        else:
            segs, weights, _ = lime_explain(model, arr, target, device, 1000)
            order = [s for s, v in sorted(weights.items(),
                                          key=lambda kv: -kv[1]) if v > 0]
            mask = np.isin(segs, order[:k]) if order else np.zeros_like(segs, bool)
            box = box_from_mask(mask, w, h, largest=False)
            sal = np.zeros(segs.shape)
            for sid, wt in weights.items():
                if wt > 0:
                    sal[segs == sid] = wt
            note = f"k {k}"
            big = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255)
                             .resize((w, h), Image.NEAREST)) > 127
            a = np.asarray(im).astype(np.float64)
            a[~big] *= 0.45                      # dim what LIME did not select
            im = Image.fromarray(a.clip(0, 255).astype(np.uint8))

        d = ImageDraw.Draw(im)
        short = min(im.size)
        wide = max(6, round(short / 48))
        # the plates are read at print size, where short/26 was unreadable
        font = _serif(max(20, round(short / 15)))
        tb, feats, _ = tebol_box(pair, stem)

        # green first, then the baseline, then TEBOL in front: where two boxes
        # nearly coincide the one behind is hidden, and that overlap is the
        # agreement the IoU plates quantify.
        for t in truths:
            d.rectangle(t, outline=bx.GREEN, width=wide)
        if box and method_box:
            d.rectangle(box, outline=BLUE, width=wide)
        if tb and method == "tebol":
            d.rectangle(tb, outline=bx.RED, width=wide)

        sc = (saliency_scores(sal, truths, w, h) if sal is not None
              else {"containment": float("nan")})
        # Each overlay shows one method against the annotation and nothing
        # else. The saliency panel carries no box, no IoU and no threshold,
        # because none of those are things the method produced -- they are
        # ours, and the paired TEBOL panel already states its own score.
        y = 6
        if method == "tebol":
            word = feats[0] if feats else "?"
            if tb:
                y = bx._tag(d, (6, y), f"TEBOL ({word})", bx.RED, font) + 5
            y = bx._tag(d, (6, y), "ImageNet ground truth", bx.GREEN, font)
            if tb:
                bx._tag(d, (6, y + 5), f"IoU {B.best_iou(tb, truths):.2f}",
                        bx.RED, font)
        else:
            y = bx._tag(d, (6, y), label, BLUE, font) + 5
            bx._tag(d, (6, y), "ImageNet ground truth", bx.GREEN, font)

        tag = (f"cont{sc['containment']:.2f}" if sal is not None
               else f"iou{B.best_iou(tb, truths):.2f}" if tb else "nobox")
        name = f"{stem}__{tag}.png"
        im.save(out_dir / name)
        print(f"  wrote {out_dir / name}")




def caption_for(pair: str, stem: str, n_words: int = 5, rep: int = 0):
    """One of the image's descriptions, as the captioner wrote it.

    An image has five at each length and the ranked words pool all of them, so
    no single description is *the* input; this returns the first, which is
    enough to show a reader what the model was working from.
    """
    f = (ROOT / "artifacts" / "captions" / pair
         / "local__qwen3-vl__32b-instruct.jsonl")
    if not f.exists():
        return None
    with f.open() as fh:
        for line in fh:
            r = json.loads(line)
            if (r.get("stem") == stem and r.get("n_words") == n_words
                    and r.get("rep") == rep):
                return r.get("caption")
    return None


def contact_sheet(method: str, out_name: str, cols: int = 3,
                  captions: bool = False) -> None:
    """Assemble the per-image overlays into one figure for the paper.

    Rows are tasks and columns alternate accurate and inaccurate cases for the
    two classes of that task, so a reader sees both outcomes for every class
    without hunting through twenty files. The panels are the overlay PNGs
    themselves, so the figure and the standalone images cannot drift apart.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })

    src = FIG_DIR / method
    files = sorted(src.glob("*.png"))
    if not files:
        sys.exit(f"no overlays in {src}; run `overlay --select ...` first")

    cls_of = {stem: cls for _, cls, stem, _, _ in images()}
    pair_of = {stem: pair for pair, _, stem, _, _ in images()}
    score = lambda f: float(f.name.split("__")[1].split(".png")[0][3:])
    panels = sorted(files, key=lambda f: cls_of.get(f.name.split("__")[0], ""))

    nrow = -(-len(panels) // cols)
    fig, axes = plt.subplots(nrow, cols,
                             figsize=(3.4 * cols,
                                      (3.4 if captions else 3.0) * nrow),
                             squeeze=False)
    for ax, f in zip(axes.ravel(), panels):
        stem = f.name.split("__")[0]
        ax.imshow(Image.open(f).convert("RGB"))
        ax.set_title(f"{cls_of.get(stem, '?')}   IoU {score(f):.2f}",
                     fontsize=15)
        if captions:
            cap = caption_for(pair_of.get(stem, ""), stem)
            if cap:
                ax.set_xlabel(f"\u201c{cap.strip().rstrip('.')}\u201d",
                              fontsize=13, style="italic", wrap=True,
                              labelpad=8)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{out_name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG_DIR / out_name}.pdf")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    r = sub.add_parser("run")
    r.add_argument("--method", choices=METHODS, required=True)
    r.add_argument("--n", type=int, default=0,
                   help="images per class; 0 means every annotated image")
    r.add_argument("--num-samples", type=int, default=1000,
                   help="LIME perturbations per image")
    r.add_argument("--no-resume", action="store_true")

    e = sub.add_parser("evaluate")
    e.add_argument("--all-images", action="store_true",
                   help="score each method on everything it has, rather than "
                        "on the images every method covers")
    c = sub.add_parser("sheet")
    c.add_argument("--method", default="tebol",
                   help="subdirectory of results/figures/pixel_baselines "
                        "holding the overlays to assemble")
    c.add_argument("--out", default=None)
    c.add_argument("--cols", type=int, default=3)
    c.add_argument("--captions", action="store_true",
                   help="print the captioner's description under each panel")

    o = sub.add_parser("overlay")
    o.add_argument("--stem", action="append")
    o.add_argument("--method", choices=(*METHODS, "tebol"),
                   default="gradcam_resnet")
    o.add_argument("--method-box", action="store_true",
                   help="also draw the box derived from the saliency map; off "
                        "by default because tau and k are our choices, not the "
                        "method's")
    o.add_argument("--select", choices=["accurate", "inaccurate", "random"],
                   help="pick examples automatically instead of --stem")
    o.add_argument("--per-class", type=int, default=1)
    o.add_argument("--tau", type=float, default=0.20)
    o.add_argument("--k", type=int, default=4)
    o.add_argument("--out", default=None)

    d = sub.add_parser("draw")
    d.add_argument("--stem", action="append", required=True,
                   help="image stem; repeat for several rows")
    d.add_argument("--tau", type=float, default=0.20,
                   help="Grad-CAM threshold for the drawn box")
    d.add_argument("--k", type=int, default=4,
                   help="LIME superpixel count for the drawn box")
    d.add_argument("--out", default="comparison")
    d.add_argument("--no-method-box", action="store_true",
                   help="draw only the annotation and the saliency map; the "
                        "derived box depends on tau or k and is not something "
                        "the method itself produced")
    d.add_argument("--methods", default="tebol,gradcam,lime_resnet,lime_vgg16",
                   help="comma-separated subset of the columns to render; "
                        "Grad-CAM alone is instant, LIME costs ~7s per image")

    args = ap.parse_args()

    if args.mode == "sheet":
        contact_sheet(args.method, args.out or f"sheet_{args.method}",
                      args.cols, args.captions)
    elif args.mode == "overlay":
        stems = args.stem or []
        if args.select:
            stems = pick_examples(args.select, args.per_class)
        if not stems:
            sys.exit("give --stem or --select")
        overlay(stems, args.tau, args.k, args.method,
                Path(args.out) if args.out else None, args.method_box)
    elif args.mode == "draw":
        draw(args.stem, args.tau, args.k, args.out,
             [m.strip() for m in args.methods.split(',') if m.strip()],
             not args.no_method_box)
    elif args.mode == "run":
        rows = sample(images(), args.n)
        run(args.method, rows, args.num_samples, not args.no_resume)
    else:
        evaluate(not args.all_images)


if __name__ == "__main__":
    main()

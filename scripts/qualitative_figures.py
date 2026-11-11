#!/usr/bin/env python3
"""The qualitative comparison figures of the paper's section "LIME vs. Grad-CAM".

Two families, both drawn from real explanation weights.

LIME_<class>      bounding box | LIME. A logistic regression on frozen VGG16
                  features of the two-class task (the "LR on VGG16 features"
                  of the accuracy table) is explained by LIME. The classifier's
                  prediction is explained, and each superpixel is tinted by the
                  weight LIME fitted for it: warm raises the probability of the
                  predicted class, cool lowers it.
GRADCAM_<class>   photograph | LIME | Grad-CAM. Both explain the same class of
                  the same network, ResNet50, and that class is stated
                  explicitly, not left to the network's top prediction.

Colours are drawn the way the data are: Grad-CAM with the jet map applied to
an RGB array, so the strongest activation is red.

    python scripts/qualitative_figures.py                # everything
    python scripts/qualitative_figures.py --only vase
    python scripts/qualitative_figures.py --family gradcam
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DATA = Path("/Users/kjzg289/Library/CloudStorage/OneDrive-AZCollaboration/"
            "study/2 semestr/Diplom/data")
RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "figures" / "qualitative"
EMB = ROOT / "artifacts" / "probe_embeddings"
SIZE = 224
SAMPLES = 1000
#: the ResNet50 figures explain a 1000-class network over about as many superpixels,
#: and 1000 perturbations left the hot pot's weights unstable across seeds (rank
#: correlation 0.30); 5000 is the budget the text LIME of the paper uses
SAMPLES_RESNET = 5000
SEED = 20260824

#: figure name -> pair and photograph. The stems are the photographs of the
#: figures in the manuscript, identified by matching their box panels against
#: the ImageNet images (thumbnail correlation, and ORB keypoints with a
#: RANSAC fit for the bee and the guitar). The box panel itself is kept as drawn
#: in the manuscript, in artifacts/qualitative/box_<figure>.png.
BOX_DIR = ROOT / "artifacts" / "qualitative"
LIME_FIGS = json.loads((ROOT / "scripts" / "qualitative_images.json")
                       .read_text(encoding="utf-8"))["lime"]
#: figure name -> (class folder, stem, ImageNet class name)
GRADCAM_FIGS = {
    "GRADCAM_hotpot": ("hotpot", "n07590611_3587", "hot pot"),
    "GRADCAM_vase": ("vase", "n04522168_2044", "vase"),
}


def _load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def find_raw(stem: str) -> Path:
    hits = list(RAW.glob(f"*/*/{stem}.JPEG"))
    if not hits:
        raise SystemExit(f"{stem}: not under {RAW}")
    return hits[0]


# --- LIME on a classifier over VGG16 features ----------------------------------

def vgg16_classifier(pair: str):
    """Standardiser + logistic regression on the cached VGG16 features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    z = np.load(EMB / f"vgg16__{pair}.npz", allow_pickle=True)
    X, stems = z["X"], [str(s) for s in z["stems"]]
    cls_of = {p.stem: p.parent.name for p in (RAW / pair).glob("*/*.JPEG")}
    y = np.array([cls_of[s] for s in stems])
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000).fit(scaler.transform(Xtr), ytr)
    return scaler, clf


def lime_on_vgg16(pair: str, stem: str, label_mode: str = "predicted"):
    import torch
    from lime import lime_image

    sv = _load(str(ROOT / "scripts" / "24_baselines.py"), "baselines")
    model, preprocess = sv.load_vgg16("cpu")
    scaler, clf = vgg16_classifier(pair)
    from PIL import Image

    img = np.asarray(Image.open(find_raw(stem)).convert("RGB").resize(
        (SIZE, SIZE)), dtype=np.uint8)

    def predict(batch):
        feats = []
        for chunk in np.array_split(np.asarray(batch, dtype=np.uint8),
                                    max(1, len(batch) // 50)):
            t = torch.stack([preprocess(Image.fromarray(a)) for a in chunk])
            with torch.no_grad():
                feats.append(model(t).numpy())
        return clf.predict_proba(scaler.transform(np.concatenate(feats)))

    p = predict(img[None])[0]
    label = int(p.argmax())
    exp = lime_image.LimeImageExplainer(random_state=SEED).explain_instance(
        img, predict, labels=(label,), hide_color=0, num_samples=SAMPLES,
        random_seed=SEED, batch_size=50)
    return img, exp, label, clf.classes_[label], float(p[label])


def heat_from(exp, label: int) -> np.ndarray:
    """Per-pixel LIME weight of the superpixel the pixel belongs to."""
    w = dict(exp.local_exp[label])
    return np.vectorize(lambda s: w.get(s, 0.0))(exp.segments)


def tint(img: np.ndarray, heat: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    import matplotlib
    from skimage.segmentation import mark_boundaries

    v = np.abs(heat).max() or 1.0
    rgb = matplotlib.colormaps["RdBu_r"]((heat / v + 1) / 2)[..., :3]
    # a grey photograph under the tint, so only the explanation carries colour
    grey = np.repeat(img.astype(np.float64).mean(axis=2, keepdims=True) / 255.0,
                     3, axis=2)
    return (1 - alpha) * grey + alpha * rgb


def lime_cached(name: str, spec: dict, refresh: bool = False):
    """img, segments, heat, class name, probability -- LIME is run once per figure."""
    BOX_DIR.mkdir(parents=True, exist_ok=True)
    f = BOX_DIR / f"lime_{name}.npz"
    if f.exists() and not refresh:
        z = np.load(f, allow_pickle=True)
        return z["img"], z["segments"], z["heat"], str(z["cname"]), float(z["prob"])
    img, exp, label, cname, prob = lime_on_vgg16(spec["pair"], spec["stem"])
    heat = heat_from(exp, label)
    np.savez_compressed(f, img=img, segments=exp.segments, heat=heat,
                        cname=cname, prob=prob)
    return img, exp.segments, heat, cname, prob


def fig_lime(name: str, spec: dict, refresh: bool = False) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from skimage.segmentation import mark_boundaries

    img, segments, heat, cname, prob = lime_cached(name, spec, refresh)
    panel = mark_boundaries(tint(img, heat), segments, color=(1, 1, 1),
                            mode="thick")
    box = Image.open(BOX_DIR / f"box_{name}.png").convert("RGB")
    fig, ax = plt.subplots(1, 2, figsize=(10, 5.2))
    ax[0].imshow(box); ax[0].set_title("Bounding box", fontsize=22)
    ax[1].imshow(panel); ax[1].set_title("LIME heatmap", fontsize=22)
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{name}: {spec['stem']}  predicted '{cname}' ({prob:.3f})  "
          f"{len(np.unique(segments))} superpixels -> {path}", flush=True)
    return path


# --- LIME and Grad-CAM on ResNet50 --------------------------------------------

def fig_gradcam(name: str, cls: str, stem: str, imagenet_name: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from lime import lime_image
    from PIL import Image
    from torchvision import models, transforms

    sg = _load(str(ROOT / "scripts" / "survey_gradcam.py"), "survey_gradcam")
    cats = models.ResNet50_Weights.IMAGENET1K_V1.meta["categories"]
    target = [i for i, c in enumerate(cats) if c.lower() == imagenet_name][0]
    model = sg.load_model()
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img_f = sg.read_image(find_raw(stem))                 # float, 224 x 224
    img_u8 = np.uint8(np.round(img_f * 255))

    def predict(batch):
        out = []
        for chunk in np.array_split(np.asarray(batch, dtype=np.uint8),
                                    max(1, len(batch) // 50)):
            t = torch.stack([norm(torch.from_numpy(a.astype(np.float32) / 255)
                                  .permute(2, 0, 1)) for a in chunk])
            with torch.no_grad():
                out.append(torch.softmax(model(t), 1).numpy())
        return np.concatenate(out)

    p = predict(img_u8[None])[0]
    exp = lime_image.LimeImageExplainer(random_state=SEED).explain_instance(
        img_u8, predict, labels=(target,), hide_color=0, num_samples=SAMPLES_RESNET,
        random_seed=SEED, batch_size=50)
    lime_panel = tint(img_u8, heat_from(exp, target))
    x = norm(torch.from_numpy(img_f).permute(2, 0, 1))[None]
    cam = sg.grad_cam(model, x, [target])[0]
    cam_panel = sg.overlay(img_f, cam)

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.3))
    for a, im, t in zip(ax, (img_f, lime_panel, cam_panel),
                        ("Original image", "LIME heatmap", "Grad-CAM heatmap")):
        a.imshow(im); a.set_title(t, fontsize=18); a.axis("off")
    plt.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{name}: {stem}  ResNet50 P({imagenet_name}) = {p[target]:.3f}, "
          f"top-1 = '{cats[int(p.argmax())]}' -> {path}", flush=True)
    return path


def extract_boxes(paper_dir: str) -> None:
    """Keep the box panel of each manuscript LIME figure, before it is replaced."""
    import subprocess
    import tempfile

    from PIL import Image

    BOX_DIR.mkdir(parents=True, exist_ok=True)
    for name in LIME_FIGS:
        pdf = Path(paper_dir) / f"{name}.pdf"
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["pdftoppm", "-png", "-r", "300", "-f", "1", "-l", "1",
                            str(pdf), f"{tmp}/p"], check=True)
            im = Image.open(next(Path(tmp).glob("p*.png"))).convert("RGB")
        g = np.asarray(im.convert("L"))
        m = g[:, :int(im.width * 0.5)] < 245
        rows = np.where(m.mean(1) > 0.5)[0]
        cols = np.where(m[rows.min():rows.max() + 1].mean(0) > 0.5)[0]
        # the first contiguous block only: a narrow photograph leaves room in the
        # left half for the edge of the neighbouring LIME panel
        cols = cols[:int(np.argmax(np.diff(cols) > 40)) + 1
                    if (np.diff(cols) > 40).any() else len(cols)]
        im.crop((cols.min(), rows.min(), cols.max() + 1, rows.max() + 1)).save(
            BOX_DIR / f"box_{name}.png")
        print(f"{name}: box panel kept ({cols.max() - cols.min() + 1} x "
              f"{rows.max() - rows.min() + 1})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=["lime", "gradcam"], action="append")
    ap.add_argument("--only", action="append", help="figure name or class")
    ap.add_argument("--refresh", action="store_true",
                    help="rerun LIME instead of using the cached weights")
    ap.add_argument("--extract-boxes", metavar="PAPER_DIR",
                    help="save the box panels of the existing LIME_*.pdf and exit")
    args = ap.parse_args()
    if args.extract_boxes:
        extract_boxes(args.extract_boxes)
        return
    fams = args.family or ["gradcam", "lime"]

    def wanted(name):
        return not args.only or any(o.lower() in name.lower() for o in args.only)

    if "gradcam" in fams:
        for name, (cls, stem, inet) in GRADCAM_FIGS.items():
            if wanted(name):
                fig_gradcam(name, cls, stem, inet)
    if "lime" in fams:
        for name, spec in LIME_FIGS.items():
            if wanted(name):
                fig_lime(name, spec, args.refresh)


if __name__ == "__main__":
    main()

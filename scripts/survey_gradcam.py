#!/usr/bin/env python3
"""Grad-CAM for the survey stimuli, once for the class ResNet50 predicts and once
for the class the image is named after.

The survey heatmaps were made by Diplom/data/test_package_data3/
MORF_for_survey.ipynb, which called `cam(input_tensor=...)` with no target.
pytorch-grad-cam then explains argmax of the model output, so a heatmap shows
why ResNet50 chose its top class, which is the named class on only 86.0% of the
1,000 stimuli. This script

  1. reproduces the survey heatmaps (target = predicted class) and compares them
     with the saved stimulus images, to show the reimplementation is faithful;
  2. computes the heatmap for the named class (target = named class);
  3. for every image reports whether the two targets agree and how far the two
     heatmaps are from each other.

Grad-CAM is implemented here (layer4 activations, spatially averaged gradients
as weights, ReLU of the weighted sum, min-max scaling, bilinear resize to
224 x 224) so the check does not depend on a package that is not installed. The
colour map is matplotlib's jet, which is the classic jet that OpenCV's
COLORMAP_JET also draws.

    python scripts/survey_gradcam.py
    python scripts/survey_gradcam.py --save-overlays      # corrected overlays
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = Path("/Users/kjzg289/Library/CloudStorage/OneDrive-AZCollaboration/"
            "study/2 semestr/Diplom/data")
CKPT = Path.home() / ".cache/torch/hub/checkpoints/resnet50-0676ba61.pth"
OUT_CSV = ROOT / "results" / "metrics" / "survey_gradcam.csv"
OUT_IMG = ROOT / "artifacts" / "survey_gradcam_corrected"

#: survey folder -> (folder with the original photographs, ImageNet class name)
CLASSES = {
    "ambulance": ("test_package_data1/ambulance", "ambulance"),
    "fire_truck": ("test_package_data1/fire_truck", "fire engine"),
    "Ant": ("test_package_data3/Ant", "ant"),
    "Bee": ("test_package_data3/Bee", "bee"),
    "Acoustic_guitar": ("test_package_data2/Acoustic_guitar", "acoustic guitar"),
    "Violin": ("test_package_data2/Violin", "violin"),
    "hotpot": ("test_package_data4/hotpot", "hot pot"),
    "vase": ("test_package_data4/vase", "vase"),
    "cucumber": ("test_package_data5/cucumber", "cucumber"),
    "zucchini": ("test_package_data5/zucchini", "zucchini"),
}
SIZE = 224


def load_model():
    import torch
    from torchvision import models

    model = models.resnet50(weights=None)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    return model.eval()


def read_image(path: Path) -> np.ndarray:
    """RGB float array in [0, 1], bilinear-resized to 224 x 224 like cv2.resize."""
    import torch
    import torch.nn.functional as F
    from PIL import Image

    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(a).permute(2, 0, 1)[None]
    t = F.interpolate(t, size=(SIZE, SIZE), mode="bilinear",
                      align_corners=False, antialias=False)
    return t[0].permute(1, 2, 0).numpy()


def grad_cam(model, x, targets):
    """Heatmaps for several targets from one forward pass. -> [len(targets), H, W]."""
    import torch
    import torch.nn.functional as F

    acts = {}

    def fwd(_m, _i, out):
        acts["a"] = out
        out.register_hook(lambda g: acts.__setitem__("g", g))

    h = model.layer4.register_forward_hook(fwd)
    maps = []
    try:
        for tgt in targets:
            model.zero_grad(set_to_none=True)
            out = model(x)
            out[0, tgt].backward()
            w = acts["g"].mean(dim=(2, 3), keepdim=True)
            cam = F.relu((w * acts["a"]).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=(SIZE, SIZE), mode="bilinear",
                                align_corners=False)[0, 0].detach().numpy()
            cam = cam - cam.min()
            maps.append(cam / (1e-7 + cam.max()))
    finally:
        h.remove()
    return np.stack(maps)


def overlay(img: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """pytorch-grad-cam's show_cam_on_image(use_rgb=True): jet + image, rescaled."""
    import matplotlib
    heat = matplotlib.colormaps["jet"](np.uint8(255 * cam))[..., :3]
    out = heat + img
    out = out / out.max()
    return np.uint8(255 * out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save-overlays", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="images per class")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from torchvision import models, transforms

    cats = models.ResNet50_Weights.IMAGENET1K_V1.meta["categories"]

    def idx(name):
        exact = [i for i, c in enumerate(cats) if c.lower() == name]
        return (exact or [i for i, c in enumerate(cats)
                          if c.lower().startswith(name)])[0]

    torch.set_num_threads(8)
    model = load_model()
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    rows = []
    for cls, (folder, name) in CLASSES.items():
        named = idx(name)
        survey_dir = DATA / "servey" / "heatmaps" / cls
        files = sorted(f for f in os.listdir(survey_dir) if f.endswith(".JPEG"))
        if args.limit:
            files = files[:args.limit]
        for k, f in enumerate(files, 1):
            src = DATA / folder / f
            if not src.exists():
                continue
            img = read_image(src)
            x = norm(torch.from_numpy(img).permute(2, 0, 1))[None]
            with torch.no_grad():
                pred = int(model(x).argmax(1))
            cam_pred, cam_named = grad_cam(model, x, [pred, named])
            saved = np.asarray(Image.open(survey_dir / f).convert("RGB"),
                               dtype=np.float32)
            mine = overlay(img, cam_pred).astype(np.float32)
            top = lambda c: c >= np.quantile(c, 0.8)   # the hottest fifth  # noqa: E731
            a, b = top(cam_pred), top(cam_named)
            rows.append({
                "class": cls, "file": f, "named": cats[named],
                "predicted": cats[pred], "match": pred == named,
                "cam_corr": float(np.corrcoef(cam_pred.ravel(),
                                              cam_named.ravel())[0, 1]),
                "top20_iou": float((a & b).sum() / (a | b).sum()),
                #: pixel distance between the saved survey image and the same
                #: heatmap recomputed here (0-255 scale)
                "repro_mad": float(np.abs(saved - mine).mean()),
            })
            if args.save_overlays:
                d = OUT_IMG / cls
                d.mkdir(parents=True, exist_ok=True)
                Image.fromarray(overlay(img, cam_named)).save(d / f, quality=95)
            if k % 25 == 0:
                print(f"  {cls}: {k}/{len(files)}", flush=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV} ({len(rows)} images)")


if __name__ == "__main__":
    main()

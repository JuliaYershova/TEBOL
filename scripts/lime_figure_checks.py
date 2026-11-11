"""Do the regenerated LIME explanations deserve trust?  Three checks per figure.

  stability   the same image explained again with two other LIME seeds: rank
              correlation of the superpixel weights, and overlap of the three
              largest positive superpixels with the first run's.
  location    how much of the positive weight lies inside the annotated object
              box (ImageNet annotation), against the share of the image the box
              covers, and whether the single largest superpixel sits in it.
  deletion    blank the three largest positive superpixels, and as a control
              three random superpixels (20 draws); compare the drop in the
              classifier's probability for the predicted class.
"""
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path.home() / "Documents" / "TEBOL"
sys.path.insert(0, str(ROOT / "src"))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


qf = load(str(ROOT / "scripts" / "qualitative_figures.py"), "qf")
B = __import__("tebol.bbox", fromlist=["x"])
sv = load(str(ROOT / "scripts" / "24_baselines.py"), "baselines")
import torch  # noqa: E402
from PIL import Image  # noqa: E402

OUT = ROOT / "results" / "metrics" / "lime_figure_checks.csv"
SEEDS = [11, 22]


def seg_weights(segments, heat):
    ids = np.unique(segments)
    return ids, np.array([heat[segments == s][0] for s in ids])


def make_predict(pair):
    model, preprocess = sv.load_vgg16("cpu")
    scaler, clf = qf.vgg16_classifier(pair)

    def predict(batch):
        feats = []
        for chunk in np.array_split(np.asarray(batch, dtype=np.uint8),
                                    max(1, len(batch) // 50)):
            t = torch.stack([preprocess(Image.fromarray(a)) for a in chunk])
            with torch.no_grad():
                feats.append(model(t).numpy())
        return clf.predict_proba(scaler.transform(np.concatenate(feats)))
    return predict


rows = []
for name, spec in qf.LIME_FIGS.items():
    pair, stem = spec["pair"], spec["stem"]
    img, segments, heat, cname, prob = qf.lime_cached(name, spec)
    ids, w0 = seg_weights(segments, heat)
    top0 = set(ids[np.argsort(-w0)[:3]])
    row = {"figure": name, "stem": stem, "predicted": cname,
           "prob": round(prob, 3), "superpixels": len(ids)}

    # --- stability: other seeds
    cors, jacs = [], []
    for sd in SEEDS:
        qf.SEED = sd
        _, exp, label, _, _ = qf.lime_on_vgg16(pair, stem)
        _, w1 = seg_weights(exp.segments, qf.heat_from(exp, label))
        cors.append(spearmanr(w0, w1).correlation)
        t1 = set(ids[np.argsort(-w1)[:3]])
        jacs.append(len(top0 & t1) / len(top0 | t1))
    row["seed_spearman"] = round(float(np.mean(cors)), 2)
    row["seed_top3_jaccard"] = round(float(np.mean(jacs)), 2)

    # --- location against the annotated box
    xmls = list((ROOT / "data" / "annotations").glob(f"*/*/{stem}.xml"))
    if xmls:
        boxes, W, H = B.truth_boxes(xmls[0])
        sx, sy = 224 / W, 224 / H
        mask = np.zeros((224, 224), bool)
        for x0, y0, x1, y1 in boxes:
            mask[int(y0 * sy):int(np.ceil(y1 * sy)), int(x0 * sx):int(np.ceil(x1 * sx))] = True
        pos = np.clip(heat, 0, None)
        row["box_area_share"] = round(float(mask.mean()), 2)
        row["positive_weight_inside_box"] = round(float(pos[mask].sum() / (pos.sum() + 1e-12)), 2)
        top = ids[int(np.argmax(w0))]
        ys, xs = np.where(segments == top)
        row["top_superpixel_centroid_in_box"] = bool(mask[int(ys.mean()), int(xs.mean())])
        row["top_superpixel_overlap_with_box"] = round(float(mask[segments == top].mean()), 2)
    else:
        row["box_area_share"] = None

    # --- deletion test
    predict = make_predict(pair)
    base = predict(img[None])[0]
    label = int(base.argmax())

    def blank(sel):
        x = img.copy()
        for s in sel:
            x[segments == s] = 0
        return predict(x[None])[0][label]

    drop_top = base[label] - blank(ids[np.argsort(-w0)[:3]])
    rng = np.random.default_rng(0)
    drops_rand = [base[label] - blank(rng.choice(ids, 3, replace=False)) for _ in range(20)]
    row["deletion_drop_top3"] = round(float(drop_top), 3)
    row["deletion_drop_random3"] = round(float(np.mean(drops_rand)), 3)
    rows.append(row)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(json.dumps(row), flush=True)

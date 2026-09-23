"""Bounding boxes asked of the captioner, conditioned on an explanation.

The question this stage answers is not "can the model find the object" -- it is
"do the words SMER says the classifier used point at the object". So the prompt
never names the class. It names the three words the explainer ranked highest
for *this* image, and asks where the thing they describe is:

    Provide bounding box coordinates for the object from (word, word, word).
    Format: x_min, y_min, x_max, y_max with integers only.

That is the Diplom prompt, kept verbatim. What comes back needs two things
doing to it before it can be scored.

**Coordinates are normalised.** Qwen3-VL answers on a 0-1000 grid whatever the
image size -- "0, 0, 999, 999" means the whole image -- while the ImageNet
ground truth is in pixels. `rescale` converts. The grid is a property of the
model, so `COORD_SPACE` is a parameter rather than a guess; a reply exceeding
it is flagged instead of silently clipped, which is how the normalisation was
caught in the first place.

**The model answers per word.** Asked about three words it often returns three
labelled boxes, one each, rather than one box for the object.

The answer is **the first box**: the words enter the prompt in SMER rank order,
so the first box is the one for the highest-ranked word -- the word the
explanation says mattered most. Measured over the first six ant_bee images,
mean IoU is 0.962 for the first box against 0.825 smallest, 0.590 union and
0.570 largest; union and largest lose because they swallow the "flower" box,
which is the whole photograph. `parse_boxes` still keeps every box, labelled
and in order, because per-word agreement is one of the evaluations the next
stage wants -- but one box per image is the answer.

A model that answers with prose, or with a degenerate box, is a parse failure
recorded as one rather than dropped, so the failure rate is itself a result.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

#: The Diplom prompt, verbatim. `features` is the comma-joined top-k words.
PROMPT = ("Provide bounding box coordinates for the object from ({features}). "
          "Format: x_min, y_min, x_max, y_max with integers only.")

#: Qwen3-VL's coordinate grid. Boxes come back on 0..1000 regardless of the
#: image's real size; see `rescale`.
COORD_SPACE = 1000

#: One run of four integers, optionally preceded by the word it belongs to:
#: "bee: 387, 88, 807, 750", "12,34,56,78", "[12, 34, 56, 78]". Negatives are
#: matched so an out-of-range answer is caught by validation rather than
#: silently re-parsed into a different, in-range box.
_LABELLED_BOX = re.compile(
    r"(?:([A-Za-z][\w'-]*)\s*[:=]\s*)?"
    r"(-?\d+)\s*[,\s]\s*(-?\d+)\s*[,\s]\s*(-?\d+)\s*[,\s]\s*(-?\d+)")


def build_prompt(features) -> str:
    """The prompt for one image, given its top words in rank order."""
    return PROMPT.format(features=", ".join(features))


def parse_boxes(text: str | None):
    """(boxes, error). boxes is [{"label": str|None, "box": [4 ints]}, ...].

    Every four-integer run in the reply, in order, each tagged with the word
    that introduced it if there was one. Coordinates are as the model gave
    them -- still on COORD_SPACE, not pixels; call `rescale`.

    Degenerate and inverted boxes are dropped with a note rather than kept,
    since a zero-area box scores IoU 0 against everything and would look like
    a confident miss instead of a bad answer.
    """
    if not text:
        return [], "empty reply"
    out, bad = [], 0
    for label, *nums in _LABELLED_BOX.findall(text):
        x0, y0, x1, y1 = (int(n) for n in nums)
        if x1 <= x0 or y1 <= y0:
            bad += 1
            continue
        out.append({"label": label.lower() or None, "box": [x0, y0, x1, y1]})
    if not out:
        return [], (f"{bad} degenerate box(es), none usable" if bad
                    else "no four-integer box in the reply")
    return out, (f"dropped {bad} degenerate box(es)" if bad else None)


def refused_labels(text: str | None) -> set[str]:
    """Words the model answered with a null box.

    Asked about a word that is not in the picture, the model does not stay
    silent -- it writes "beetle: 0, 0, 0, 0" or "beetle: -1, -1, -1, -1". That
    is a denial, and it is a different thing from a word the model merged into
    another box or never mentioned. `parse_boxes` drops these as degenerate,
    which is right for scoring and wrong for attribution, so they are
    recovered here.
    """
    out = set()
    for label, *nums in _LABELLED_BOX.findall(text or ""):
        x0, y0, x1, y1 = (int(n) for n in nums)
        if label and (x1 <= x0 or y1 <= y0):
            out.add(label.lower())
    return out


def rescale(box, width: int, height: int, space: int = COORD_SPACE):
    """A box on the 0..`space` grid, converted to pixels. (box, out_of_range).

    `out_of_range` flags a coordinate beyond the grid -- a sign the model used
    a different convention, which should be investigated rather than clipped
    away.
    """
    over = any(c < 0 or c > space for c in box)
    x0, y0, x1, y1 = box
    px = [round(x0 * width / space), round(y0 * height / space),
          round(x1 * width / space), round(y1 * height / space)]
    return px, over


def clip(box, width: int, height: int):
    """Box clipped to the image. Returns (box, was_clipped)."""
    x0, y0, x1, y1 = box
    c = [max(0, min(x0, width)), max(0, min(y0, height)),
         max(0, min(x1, width)), max(0, min(y1, height))]
    return c, c != list(box)


def truth_boxes(xml_path: str | Path):
    """(boxes, width, height) from an ImageNet annotation.

    An image may carry several objects -- 4,914 XML files cover 4,531 images --
    so this returns every box and leaves the choice of which to score to the
    caller.
    """
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w = int(size.findtext("width"))
    h = int(size.findtext("height"))
    boxes = []
    for obj in root.findall("object"):
        b = obj.find("bndbox")
        boxes.append([int(b.findtext("xmin")), int(b.findtext("ymin")),
                      int(b.findtext("xmax")), int(b.findtext("ymax"))])
    return boxes, w, h


def iou(a, b) -> float:
    """Intersection over union of two [x_min, y_min, x_max, y_max] boxes."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def best_iou(pred, truths) -> float:
    """IoU against the best-matching ground-truth box, 0.0 if there are none."""
    return max((iou(pred, t) for t in truths), default=0.0)


def union(boxes):
    """The smallest box containing all of them -- one reading of a per-word reply."""
    if not boxes:
        return None
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]

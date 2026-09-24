"""The five arms of the stage-3 comparison, and the matrices they load.

An *arm* is one answer to "what text does the classifier see". They exist to
separate three effects that the naive setup confounds:

    caption                  the descriptions, as generated
    caption_noclass          the same descriptions with the class name removed
    tags                     the human ImageNet-Captions tags
    caption_tagsub           descriptions, on the images tags cover
    caption_noclass_tagsub   ditto, class name removed

`caption` vs `caption_noclass` measures leakage: 85% of captions name their own
class, so a classifier scoring 0.99 may be reading the label off the caption.
`tags` vs `caption_tagsub` is the reviewers' question -- are descriptions
better than tags -- and it has to run on the tag-covered images, because tags
exist for only a third of the corpus and comparing across different image sets
would confound representation with sample. `caption` vs `caption_tagsub` prices
that restriction, so the two comparisons can be read together.

There is deliberately no `tags_noclass`. Tags are what a human wrote about the
photograph; stripping the class word from them would answer a question nobody
asked, and the tag arm exists as the human baseline, not as an ablation.

**Setups** are caption lengths: stage 1 asked for 3, 5, 7, 10, 15, 20, 25 and
30 words. Each is trained separately, so caption length is a variable of the
study rather than a constant nobody chose. Tags have no length -- their index
stores n_words=0 -- so the tag arm has the single setup 0.

The requested length is a request, not a constraint: the model overshoots at
every setting (asked 3, got 3.71 words on average; asked 30, got 34.40), and
the rate at which a caption names its own class climbs with length, from 79.0%
at 3 words to 90.0% at 30. Both are reported in
results/metrics/class_leakage.md, and both are why length is a variable here.

The heavy lifting is stage 2's: `vocab.npz` holds one vector per distinct word,
and `index/<pair>.npz` says which vocab rows each caption uses. This module
filters that index to one setup, drops words whose embedding call failed, and
returns the CSR arrays plus the vocab matrix. It does not build the design
matrix -- see `caption_means` -- because SMER needs the per-word rows and the
design matrix is a mean over them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .vectors_io import MISSING, VectorStore, load_index, load_matrix

CAPTION_SETUPS = (3, 5, 7, 10, 15, 20, 25, 30)
TAG_SETUP = 0

EMB_FULL = "local__qwen3-embedding__4b"
EMB_NOCLASS = "local__qwen3-embedding__4b--noclass"


@dataclass(frozen=True)
class Arm:
    """One column of the comparison table."""

    name: str
    embedding: str      #: directory under artifacts/embeddings
    index: str          #: "index" (captions) or "index_tags" (tags)
    setups: tuple[int, ...]
    tag_subset: bool    #: restrict to images ImageNet-Captions covers

    @property
    def matrix(self) -> str:
        return "tags.npz" if self.index == "index_tags" else "vocab.npz"


ARMS: dict[str, Arm] = {
    a.name: a for a in (
        Arm("caption", EMB_FULL, "index", CAPTION_SETUPS, False),
        Arm("caption_noclass", EMB_NOCLASS, "index", CAPTION_SETUPS, False),
        Arm("tags", EMB_FULL, "index_tags", (TAG_SETUP,), False),
        Arm("caption_tagsub", EMB_FULL, "index", CAPTION_SETUPS, True),
        Arm("caption_noclass_tagsub", EMB_NOCLASS, "index", CAPTION_SETUPS, True),
    )
}

#: Tag coverage is defined by the *full* embedding run: the noclass run has no
#: index_tags of its own, and tags are never class-stripped, so the noclass
#: tag-subset arm takes its captions from the noclass vocab and its image list
#: from here.
TAG_INDEX_SOURCE = EMB_FULL


@dataclass
class ArmData:
    """One (pair, arm, setup), ready to train and to explain.

    `rows` / `offsets` are CSR over `store.vectors`: text i owns
    `rows[offsets[i]:offsets[i + 1]]`, in caption order, with failed embeddings
    already dropped. Everything downstream -- the design matrix, the per-word
    logits, the AOPC table -- is a reduction over those two arrays.
    """

    pair: str
    arm: str
    setup: int
    stem: np.ndarray     #: [n_texts] image id
    rep: np.ndarray      #: [n_texts] repetition index (0 for tags)
    cls: np.ndarray      #: [n_texts] class label
    rows: np.ndarray     #: [n_slots] vocab row per word occurrence
    offsets: np.ndarray  #: [n_texts + 1]
    store: VectorStore

    def __len__(self) -> int:
        return len(self.stem)

    @property
    def lengths(self) -> np.ndarray:
        """Words per text, after dropping failed embeddings."""
        return np.diff(self.offsets)

    def word_rows(self, i: int) -> np.ndarray:
        return self.rows[self.offsets[i]:self.offsets[i + 1]]

    def words(self, i: int) -> list[str]:
        return [self.store.keys[r] for r in self.word_rows(i)]


def tag_stems(root: Path, pair: str) -> set[str]:
    """Images ImageNet-Captions gives tags for."""
    path = (root / "artifacts" / "embeddings" / TAG_INDEX_SOURCE
            / "index_tags" / f"{pair}.npz")
    return {str(s) for s in load_index(path).stem}


def load_arm(root: Path, pair: str, arm_name: str, setup: int) -> ArmData:
    """Load one (pair, arm, setup) and rebuild its CSR without MISSING slots."""
    arm = ARMS[arm_name]
    if setup not in arm.setups:
        raise ValueError(f"arm {arm_name!r} has no setup {setup}")

    emb = root / "artifacts" / "embeddings" / arm.embedding
    index = load_index(emb / arm.index / f"{pair}.npz")
    store = load_matrix(emb / arm.matrix)

    keep = np.ones(len(index), dtype=bool)
    if arm.index == "index":
        keep &= index.n_words == setup
    if arm.tag_subset:
        allowed = tag_stems(root, pair)
        keep &= np.array([str(s) in allowed for s in index.stem])

    sel = np.flatnonzero(keep)
    flat: list[np.ndarray] = []
    offsets = np.zeros(len(sel) + 1, dtype=np.int64)
    for j, i in enumerate(sel):
        r = index.rows[index.offsets[i]:index.offsets[i + 1]]
        r = r[r != MISSING]
        flat.append(r)
        offsets[j + 1] = offsets[j] + len(r)

    rows = (np.concatenate(flat) if flat
            else np.empty(0, dtype=np.int32)).astype(np.int32)

    return ArmData(
        pair=pair, arm=arm_name, setup=setup,
        stem=np.array([str(s) for s in index.stem[sel]]),
        rep=index.rep[sel].astype(np.int16),
        cls=np.array([str(c) for c in index.cls[sel]]),
        rows=rows, offsets=offsets, store=store,
    )


def merge_arms(parts: list[ArmData]) -> ArmData:
    """Concatenate arms loaded from different pairs.

    Safe because `rows` indexes the one global vocab table, so a word id means
    the same thing in every pair; only the CSR offsets need rebasing.
    """
    rows = np.concatenate([p.rows for p in parts]) if parts else np.empty(0, np.int32)
    offsets, base = [np.zeros(1, dtype=np.int64)], 0
    for p in parts:
        offsets.append(p.offsets[1:] + base)
        base += len(p.rows)
    return ArmData(
        pair="+".join(dict.fromkeys(p.pair for p in parts)),
        arm=parts[0].arm, setup=parts[0].setup,
        stem=np.concatenate([p.stem for p in parts]),
        rep=np.concatenate([p.rep for p in parts]),
        cls=np.concatenate([p.cls for p in parts]),
        rows=rows.astype(np.int32),
        offsets=np.concatenate(offsets), store=parts[0].store,
    )


def load_classes(root: Path, classes: Sequence[str], arm_name: str,
                 setup: int) -> ArmData:
    """One design matrix over any set of classes, from whichever pairs hold them.

    The multiclass experiments need classes that live in different pair files;
    this loads each pair once, keeps only the wanted classes and the texts that
    kept a word, and merges. Captions with no embedded word are dropped here
    rather than left as zero rows for the caller to notice.
    """
    where = {c.name: p.name for p in sorted((root / "data" / "raw").iterdir())
             if p.is_dir() for c in sorted(p.iterdir()) if c.is_dir()}
    missing = [c for c in classes if c not in where]
    if missing:
        raise ValueError(f"no images for {missing}; have {sorted(where)}")

    parts = []
    for pair in dict.fromkeys(where[c] for c in classes):
        d = load_arm(root, pair, arm_name, setup)
        sel = np.flatnonzero(np.isin(d.cls, list(classes)) & nonempty(d))
        flat = [d.rows[d.offsets[i]:d.offsets[i + 1]] for i in sel]
        off = np.zeros(len(sel) + 1, dtype=np.int64)
        off[1:] = np.cumsum([len(f) for f in flat])
        parts.append(ArmData(
            pair=pair, arm=arm_name, setup=setup,
            stem=d.stem[sel], rep=d.rep[sel], cls=d.cls[sel],
            rows=(np.concatenate(flat) if flat
                  else np.empty(0, np.int32)).astype(np.int32),
            offsets=off, store=d.store))
    return merge_arms(parts)


def caption_means(data: ArmData) -> np.ndarray:
    """[n_texts, dim] -- the mean-pooled vectors the classifier is fit on.

    Segmented mean over the CSR, not a Python loop: 12k captions x 2560 dims is
    a fraction of a second this way and about a minute the obvious way.

    A text whose words all failed to embed gets a zero row. That is a real
    input, not a sentinel, so callers should drop such rows before fitting --
    `nonempty` returns the mask.
    """
    lengths = data.lengths
    dim = data.store.dim
    out = np.zeros((len(data), dim), dtype=np.float32)
    good = np.flatnonzero(lengths > 0)
    if good.size:
        starts = data.offsets[:-1][good]
        sums = np.add.reduceat(data.store.vectors[data.rows], starts, axis=0)
        out[good] = sums / lengths[good][:, None]
    return out


def nonempty(data: ArmData) -> np.ndarray:
    """Texts that kept at least one embedded word."""
    return data.lengths > 0

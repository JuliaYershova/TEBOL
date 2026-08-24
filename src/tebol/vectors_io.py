"""Reading and writing the stage-2 word vectors.

Two files make up a run, both under artifacts/embeddings/<backend>__<model>/:

    vocab.npz          float32 [n_words, dim], one row per distinct word
    index/<pair>.npz   for every (stem, n_words, rep): the vocab rows of that
                       caption's words, in caption order

The split exists so the expensive part is stored once. A word's vector does not
depend on which caption it appeared in, so `vocab.npz` holds each word once and
`index/<pair>.npz` is the join back to captions -- 1.7M word occurrences become
1.7M int32s instead of 1.7M float32 vectors (17 GB saved).

Ragged rows are stored CSR-style: a flat `rows` array plus `offsets`, so
caption i owns rows[offsets[i]:offsets[i + 1]]. A row of -1 means the word is
not in the vocabulary, which happens only when its embedding call failed; a
rerun of stage 2 fills it in.

Aggregation is **mean**, matching the Diplom notebook (`np.mean(axis=0)` over
the caption's word embeddings). It is recorded in meta.json rather than left to
each downstream notebook, because sum and mean give different SMER scores and
the choice has to be made once. The mean still decomposes additively -- each
word contributes w.e(word)/n -- which is what SMER needs.

Deliberate departures from the Diplom pipeline, both decided upstream:

    tokenising   the notebook used bare str.split(), so 'dill.' and 'dill' were
                 separate features; stage 1 uses tebol.text.tokenize instead
    dedup        the notebook called the API once per word *occurrence*; a word
                 embeds the same wherever it occurs, so distinct words are
                 enough, at 0.6% of the calls

The stored vectors are unit length as the server returns them, and reproducible
to ~1e-3 per component rather than exactly: batch composition shifts the result
slightly (see tebol.embedders.base). Compare directions, not bytes.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

VOCAB_NPZ = "vocab.npz"
VOCAB_CSV = "vocab.csv"
TAGS_NPZ = "tags.npz"
TAGS_CSV = "tags.csv"
META_JSON = "meta.json"
INDEX_DIR = "index"
#: image -> ImageNet-Captions tag rows. Same CSR layout as INDEX_DIR, but
#: n_words and rep are unused (written as 0): a tag set belongs to an image,
#: not to a caption generated at some length.
INDEX_TAGS_DIR = "index_tags"

MISSING = -1  # a word whose embedding call failed


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------

def atomic_write(path: Path, write) -> None:
    """Write via a temp file in the same directory, then rename.

    A run killed mid-write leaves the previous good file untouched rather than
    a half-written matrix that loads without complaint and is wrong.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("wb") as fh:
            write(fh)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def save_matrix(path: Path, keys: Sequence[str], vectors: np.ndarray) -> None:
    """Store the [n, dim] matrix and the key list whose order it follows."""
    if len(keys) != len(vectors):
        raise ValueError(f"{len(keys)} keys but {len(vectors)} vectors")
    atomic_write(path, lambda fh: np.savez(
        fh, keys=np.asarray(keys, dtype=object), vectors=vectors))


def save_sidecar(path: Path, rows: Iterable[dict], fields: Sequence[str]) -> None:
    """The human-readable twin of the matrix -- row order is the matrix order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fields))
        w.writeheader()
        w.writerows(rows)


def save_meta(dirpath: Path, meta: dict) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / META_JSON).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_meta(dirpath: Path) -> dict:
    path = Path(dirpath) / META_JSON
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_index(path: Path, records: Sequence[dict], vocab_rows: dict[str, int],
               meta: dict | None = None) -> dict:
    """Write the caption -> vocab-row join for one pair.

    `records` are dicts with stem / n_words / rep / class / tokens. Returns
    counts so the caller can report how many word slots had no vector.
    """
    offsets = np.zeros(len(records) + 1, dtype=np.int64)
    flat: list[int] = []
    for i, rec in enumerate(records):
        for tok in rec["tokens"]:
            flat.append(vocab_rows.get(tok, MISSING))
        offsets[i + 1] = len(flat)

    rows = np.asarray(flat, dtype=np.int32)
    payload = {
        "stem": np.asarray([r["stem"] for r in records], dtype=object),
        "n_words": np.asarray([r["n_words"] for r in records], dtype=np.int16),
        "rep": np.asarray([r["rep"] for r in records], dtype=np.int16),
        "cls": np.asarray([r["class"] for r in records], dtype=object),
        "rows": rows,
        "offsets": offsets,
        "meta": np.asarray(json.dumps(meta or {}), dtype=object),
    }
    atomic_write(path, lambda fh: np.savez(fh, **payload))
    return {"captions": len(records), "slots": int(rows.size),
            "missing": int((rows == MISSING).sum())}


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------

@dataclass
class VectorStore:
    """A matrix plus the keys addressing it. Built by `load_matrix`."""

    keys: list[str]
    vectors: np.ndarray  # float32 [n, dim]
    meta: dict

    def __post_init__(self) -> None:
        self.row = {k: i for i, k in enumerate(self.keys)}

    def __len__(self) -> int:
        return len(self.keys)

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def vec(self, key: str) -> np.ndarray | None:
        i = self.row.get(key)
        return None if i is None else self.vectors[i]

    def matrix(self, keys: Sequence[str]) -> np.ndarray:
        """[len(keys), dim] -- the per-word embeddings of one caption, in order.

        Unknown keys are dropped rather than zero-filled: a zero row would be a
        silent extra term in the mean and pull every score toward nothing.
        """
        idx = [self.row[k] for k in keys if k in self.row]
        if not idx:
            return np.empty((0, self.dim), dtype=self.vectors.dtype)
        return self.vectors[idx]

    def caption_vector(self, tokens: Sequence[str],
                       aggregation: str = "mean") -> np.ndarray:
        """Aggregate a caption's word vectors. 'mean' matches the Diplom run.

        Not re-normalised: the mean of unit vectors is not a unit vector, and
        normalising it would rescale every SMER contribution by a per-caption
        constant for no gain.
        """
        m = self.matrix(tokens)
        if m.shape[0] == 0:
            return np.zeros(self.dim, dtype=self.vectors.dtype)
        if aggregation == "mean":
            return m.mean(axis=0)
        if aggregation == "sum":
            return m.sum(axis=0)
        raise ValueError(f"unknown aggregation {aggregation!r} (mean | sum)")


def load_matrix(path: str | Path, meta: dict | None = None) -> VectorStore:
    """Load vocab.npz / tags.npz. `meta` defaults to the sibling meta.json."""
    path = Path(path)
    with np.load(path, allow_pickle=True) as z:
        keys = [str(k) for k in z["keys"]]
        vectors = z["vectors"]
    return VectorStore(keys=keys, vectors=vectors,
                       meta=meta if meta is not None else load_meta(path.parent))


def load_vocab(dirpath: str | Path) -> VectorStore:
    return load_matrix(Path(dirpath) / VOCAB_NPZ)


@dataclass
class CaptionIndex:
    """The caption -> vocab-row join for one pair, as written by `save_index`."""

    stem: np.ndarray
    n_words: np.ndarray
    rep: np.ndarray
    cls: np.ndarray
    rows: np.ndarray
    offsets: np.ndarray
    meta: dict

    def __len__(self) -> int:
        return len(self.stem)

    def word_rows(self, i: int, drop_missing: bool = True) -> np.ndarray:
        """Vocab rows for caption i, in caption order."""
        r = self.rows[self.offsets[i]:self.offsets[i + 1]]
        return r[r != MISSING] if drop_missing else r

    def word_vectors(self, i: int, store: VectorStore) -> np.ndarray:
        """[n_words_in_caption, dim] -- what stage 3 actually consumes."""
        return store.vectors[self.word_rows(i)]

    def caption_vector(self, i: int, store: VectorStore,
                       aggregation: str = "mean") -> np.ndarray:
        m = self.word_vectors(i, store)
        if m.shape[0] == 0:
            return np.zeros(store.dim, dtype=store.vectors.dtype)
        return m.mean(axis=0) if aggregation == "mean" else m.sum(axis=0)

    def key(self, i: int) -> tuple[str, int, int]:
        return str(self.stem[i]), int(self.n_words[i]), int(self.rep[i])


def load_index(path: str | Path) -> CaptionIndex:
    with np.load(Path(path), allow_pickle=True) as z:
        return CaptionIndex(
            stem=z["stem"], n_words=z["n_words"], rep=z["rep"], cls=z["cls"],
            rows=z["rows"], offsets=z["offsets"],
            meta=json.loads(str(z["meta"])),
        )


# --------------------------------------------------------------------------
# per-image view
# --------------------------------------------------------------------------

@dataclass
class ImageVectors:
    """Every caption of one image, grouped so a setup is one lookup.

    Written per image by scripts/02_export_word_vectors.py. Holds vocab rows,
    not vectors -- a word's vector lives once in vocab.npz, and duplicating it
    per occurrence would turn 50 MB of ids into 29 GB of floats. `words` is
    stored alongside so the file is readable without loading the matrix.
    """

    stem: str
    cls: str
    pair: str
    n_words: np.ndarray
    rep: np.ndarray
    words: np.ndarray   # flat, parallel to rows
    rows: np.ndarray    # flat vocab rows
    offsets: np.ndarray

    def __len__(self) -> int:
        return len(self.n_words)

    def setups(self) -> list[int]:
        """Word counts present, e.g. [3, 5, 7, 10]."""
        return sorted({int(n) for n in self.n_words})

    def get(self, n_words: int, rep: int,
            store: VectorStore) -> tuple[list[str], np.ndarray]:
        """(words, [n_words_in_caption, dim]) for one caption of this image."""
        hit = np.flatnonzero((self.n_words == n_words) & (self.rep == rep))
        if not hit.size:
            raise KeyError(f"{self.stem}: no caption with n_words={n_words} rep={rep}")
        i = int(hit[0])
        lo, hi = int(self.offsets[i]), int(self.offsets[i + 1])
        rows = self.rows[lo:hi]
        keep = rows != MISSING
        return [str(w) for w in self.words[lo:hi][keep]], store.vectors[rows[keep]]

    def setup(self, n_words: int,
              store: VectorStore) -> dict[int, tuple[list[str], np.ndarray]]:
        """{rep: (words, vectors)} -- every repetition at one word count."""
        reps = sorted(int(self.rep[i]) for i in
                      np.flatnonzero(self.n_words == n_words))
        return {r: self.get(n_words, r, store) for r in reps}


def save_image_vectors(path: Path, stem: str, cls: str, pair: str,
                       captions: Sequence[dict]) -> None:
    """`captions` are dicts with n_words / rep / words / rows."""
    offsets = np.zeros(len(captions) + 1, dtype=np.int64)
    words: list[str] = []
    rows: list[int] = []
    for i, c in enumerate(captions):
        words.extend(c["words"])
        rows.extend(c["rows"])
        offsets[i + 1] = len(rows)
    payload = {
        "stem": np.asarray(stem, dtype=object),
        "cls": np.asarray(cls, dtype=object),
        "pair": np.asarray(pair, dtype=object),
        "n_words": np.asarray([c["n_words"] for c in captions], dtype=np.int16),
        "rep": np.asarray([c["rep"] for c in captions], dtype=np.int16),
        "words": np.asarray(words, dtype=object),
        "rows": np.asarray(rows, dtype=np.int32),
        "offsets": offsets,
    }
    atomic_write(path, lambda fh: np.savez(fh, **payload))


def load_image_vectors(path: str | Path) -> ImageVectors:
    with np.load(Path(path), allow_pickle=True) as z:
        return ImageVectors(
            stem=str(z["stem"]), cls=str(z["cls"]), pair=str(z["pair"]),
            n_words=z["n_words"], rep=z["rep"], words=z["words"],
            rows=z["rows"], offsets=z["offsets"],
        )

"""The Embedder interface and batch driver.

Stage 2 embeds *words*, not captions. SMER explains a caption by splitting the
classifier's logit into one contribution per word, which only works because the
caption vector is a linear function of its word vectors:

    logit = w . (1/n) SUM_i e(word_i) + b
          =       (1/n) SUM_i (w . e(word_i)) + b

So the per-word vector is the primitive and the caption vector is assembled from
it downstream -- see tebol.vectors_io. That also makes LIME affordable: dropping
words from a caption is arithmetic on cached vectors, not thousands of API calls.

Two consequences worth stating, because they look like shortcuts and are not:

    dedup      A word embeds to the same vector wherever it occurs, so the 10.7k
               distinct words carry what all 1.9M occurrences would, for 0.6% of
               the calls. tebol.vectors_io holds the join back to captions.
    batching   The API takes up to 1024 inputs per call. Order is *not* promised
               to match the input, so openai_compat sorts on the response index.

Measured against litellm.vse.cz, the endpoint is deterministic *for a fixed
batch* -- the same batch twice is bit-identical -- but batch composition
perturbs the result: five words alone versus the same five inside a batch of
1024 differ by up to 1.9e-3 per component (cosine >= 0.9999). That is vLLM's
batched reduction order, not sampling. It is far below the noise any downstream
metric cares about, but it means vectors are only reproducible when the batch
size is, so a checker must compare directions, never exact bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

#: The API accepts more, but a failed batch is re-tried by halving, and a
#: too-large batch makes each failure expensive. 1024 measured at ~340 texts/s.
DEFAULT_BATCH = 1024


@dataclass(frozen=True)
class EmbedResult:
    """One API call. `vectors` is None if the call failed after its retries."""
    vectors: list[list[float]] | None
    error: str | None = None
    prompt_tokens: int | None = None
    #: seconds for the call that produced this result (excludes retry backoff)
    latency_s: float | None = None
    #: how many attempts were made, 1 when it worked first time
    attempts: int = 1


class Embedder(Protocol):
    """Anything that turns a batch of strings into a batch of vectors."""

    name: str

    def embed(self, texts: Sequence[str]) -> EmbedResult:
        ...


def embed_all(
    embedder: Embedder,
    texts: Sequence[str],
    batch_size: int = DEFAULT_BATCH,
    progress: Callable[[int, int, dict], None] | None = None,
) -> tuple[list[list[float] | None], dict[int, str], dict]:
    """Embed every text, returning vectors aligned with `texts`.

    A batch that fails is split in half and each half retried, down to single
    texts. One over-long or malformed word therefore costs its own slot rather
    than the 1,023 words that happened to share its batch -- which matters
    because the vocabulary is built from model output, not from a fixed list.

    Returns (vectors, errors, stats): `vectors[i]` is None exactly when
    `errors[i]` explains why, so a partial run is still usable and a rerun
    picks up only what is missing.
    """
    vectors: list[list[float] | None] = [None] * len(texts)
    errors: dict[int, str] = {}
    stats = {"calls": 0, "splits": 0, "retries": 0,
             "prompt_tokens": 0, "latency_s": 0.0}

    def run(lo: int, hi: int) -> None:
        res = embedder.embed(texts[lo:hi])
        stats["calls"] += 1
        stats["retries"] += res.attempts - 1
        stats["latency_s"] += res.latency_s or 0.0

        if res.vectors is not None:
            stats["prompt_tokens"] += res.prompt_tokens or 0
            for i, vec in enumerate(res.vectors):
                vectors[lo + i] = vec
            return

        if hi - lo == 1:
            errors[lo] = res.error or "unknown error"
            return

        stats["splits"] += 1
        mid = (lo + hi) // 2
        run(lo, mid)
        run(mid, hi)

    for lo in range(0, len(texts), batch_size):
        hi = min(lo + batch_size, len(texts))
        run(lo, hi)
        if progress:
            progress(hi, len(texts), stats)

    return vectors, errors, stats

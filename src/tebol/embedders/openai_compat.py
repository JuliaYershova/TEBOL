"""Embedder for any OpenAI-compatible /v1/embeddings endpoint.

Same shape as the captioner: api.openai.com and the school's litellm gateway
speak the same protocol, so only base_url and the model name differ.
"""

from __future__ import annotations

import time
from typing import Sequence

from .base import EmbedResult


class OpenAICompatEmbedder:
    def __init__(self, model: str, api_key: str, base_url: str | None = None,
                 max_retries: int = 4):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("pip install openai") from exc

        self.model = model
        self.name = f"{'local' if base_url else 'openai'}:{model}"
        self.max_retries = max_retries
        self.client = OpenAI(api_key=api_key, base_url=base_url or None)

    def embed(self, texts: Sequence[str]) -> EmbedResult:
        if not texts:
            return EmbedResult(vectors=[], prompt_tokens=0, latency_s=0.0)

        latency = 0.0
        attempt = 0
        last = "unknown error"
        for attempt in range(self.max_retries):
            t0 = time.perf_counter()
            try:
                r = self.client.embeddings.create(model=self.model,
                                                  input=list(texts))
                latency = time.perf_counter() - t0

                # the response carries an explicit index and is not promised to
                # come back in request order -- sorting on it is the difference
                # between a correct vocabulary and a silently shuffled one
                data = sorted(r.data, key=lambda d: d.index)
                if len(data) != len(texts):
                    return EmbedResult(
                        vectors=None,
                        error=f"asked for {len(texts)} vectors, got {len(data)}",
                        latency_s=round(latency, 3), attempts=attempt + 1)

                vectors = [d.embedding for d in data]
                dims = {len(v) for v in vectors}
                if len(dims) != 1:
                    return EmbedResult(
                        vectors=None, error=f"ragged dimensions: {sorted(dims)}",
                        latency_s=round(latency, 3), attempts=attempt + 1)

                usage = getattr(r, "usage", None)
                return EmbedResult(
                    vectors=vectors,
                    prompt_tokens=getattr(usage, "prompt_tokens", None),
                    latency_s=round(latency, 3),
                    attempts=attempt + 1,
                )
            except Exception as exc:
                latency = time.perf_counter() - t0
                last = f"{type(exc).__name__}: {exc}"
                # these will not fix themselves on retry; embed_all splits the
                # batch instead, which isolates the one text actually at fault
                if any(s in last.lower() for s in
                       ("invalid_api_key", "authentication", "not found",
                        "unsupported", "invalid_request", "too long",
                        "maximum context", "context length")):
                    break
                time.sleep(min(2 ** attempt, 30))

        return EmbedResult(vectors=None, error=last,
                           latency_s=round(latency, 3), attempts=attempt + 1)

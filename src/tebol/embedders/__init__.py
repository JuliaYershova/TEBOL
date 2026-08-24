"""Embedder backends. `get_embedder` picks one from config + environment."""

from __future__ import annotations

import os

from .base import DEFAULT_BATCH, EmbedResult, Embedder, embed_all

__all__ = ["get_embedder", "Embedder", "EmbedResult", "embed_all",
           "DEFAULT_BATCH"]


def get_embedder(backend: str, model: str | None = None, **kw):
    """backend: 'local' (school litellm gateway) | 'openai' (api.openai.com)."""
    from .openai_compat import OpenAICompatEmbedder

    if backend == "openai":
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise SystemExit("OPENAI_API_KEY not set -- put it in .env")
        return OpenAICompatEmbedder(
            model=model or os.environ.get("OPENAI_MODEL_EMBED",
                                          "text-embedding-3-small"),
            api_key=key, **kw)

    if backend == "local":
        base_url = os.environ.get("LOCAL_OPENAI_BASE_URL", "")
        if not base_url:
            raise SystemExit("LOCAL_OPENAI_BASE_URL not set -- put it in .env")
        model = model or os.environ.get("LOCAL_MODEL_EMBED", "")
        if not model:
            raise SystemExit(
                "no model given -- pass --model or set LOCAL_MODEL_EMBED in .env")
        return OpenAICompatEmbedder(
            model=model,
            # many local servers ignore the key but the client requires one
            api_key=os.environ.get("LOCAL_OPENAI_API_KEY") or "EMPTY",
            base_url=base_url, **kw)

    raise SystemExit(f"unknown backend {backend!r} (local | openai)")

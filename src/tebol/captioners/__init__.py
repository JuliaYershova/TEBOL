"""Captioner backends. `get_captioner` picks one from config + environment."""

from __future__ import annotations

import os

from .base import (USER_PROMPT, CaptionResult, Captioner, build_prompt,
                   max_tokens_for)

__all__ = ["get_captioner", "Captioner", "CaptionResult", "build_prompt",
           "max_tokens_for", "USER_PROMPT"]


def get_captioner(backend: str, model: str | None = None, **kw):
    """backend: 'local' (school Qwen server) | 'openai' (api.openai.com)."""
    from .openai_compat import OpenAICompatCaptioner

    if backend == "openai":
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise SystemExit("OPENAI_API_KEY not set -- put it in .env")
        return OpenAICompatCaptioner(
            model=model or os.environ.get("OPENAI_MODEL_VISION", "gpt-4o-mini"),
            api_key=key, **kw)

    if backend == "local":
        base_url = os.environ.get("LOCAL_OPENAI_BASE_URL", "")
        if not base_url:
            raise SystemExit("LOCAL_OPENAI_BASE_URL not set -- put it in .env")
        model = model or os.environ.get("LOCAL_MODEL_VISION", "")
        if not model:
            raise SystemExit(
                "no model given -- pass --model or set LOCAL_MODEL_VISION in .env")
        return OpenAICompatCaptioner(
            model=model,
            # many local servers ignore the key but the client requires one
            api_key=os.environ.get("LOCAL_OPENAI_API_KEY") or "EMPTY",
            base_url=base_url, **kw)

    raise SystemExit(f"unknown backend {backend!r} (local | openai)")

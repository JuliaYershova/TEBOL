"""Captioner for any OpenAI-compatible chat endpoint.

Covers both api.openai.com and the school's local Qwen server -- they speak the
same protocol, so only base_url and the model name differ.
"""

from __future__ import annotations

import time
from pathlib import Path

from .base import CaptionResult, build_prompt, encode_image, max_tokens_for


class OpenAICompatCaptioner:
    def __init__(self, model: str, api_key: str, base_url: str | None = None,
                 max_retries: int = 4, image_mime: str = "image/jpeg"):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("pip install openai") from exc

        self.model = model
        self.name = f"{'local' if base_url else 'openai'}:{model}"
        self.max_retries = max_retries
        self.image_mime = image_mime
        self.client = OpenAI(api_key=api_key, base_url=base_url or None)

    def caption(self, image_path: str | Path, n_words: int,
                temperature: float, seed: int | None = None) -> CaptionResult:
        # encoding failures must not abort a multi-hour run over one bad image
        try:
            b64 = encode_image(image_path)
        except Exception as exc:
            return CaptionResult(text=None,
                                 error=f"encode: {type(exc).__name__}: {exc}")

        # single user message, no system instruction -- as in Diplom
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": build_prompt(n_words)},
                {"type": "image_url",
                 "image_url": {"url": f"data:{self.image_mime};base64,{b64}"}},
            ],
        }]

        kwargs = dict(model=self.model, messages=messages,
                      max_tokens=max_tokens_for(n_words),
                      temperature=temperature)
        if seed is not None:
            kwargs["seed"] = seed

        last = "unknown error"
        for attempt in range(self.max_retries):
            t0 = time.perf_counter()
            try:
                r = self.client.chat.completions.create(**kwargs)
                latency = time.perf_counter() - t0
                choice = r.choices[0]
                usage = getattr(r, "usage", None)
                return CaptionResult(
                    text=(choice.message.content or "").strip(),
                    finish_reason=getattr(choice, "finish_reason", None),
                    prompt_tokens=getattr(usage, "prompt_tokens", None),
                    completion_tokens=getattr(usage, "completion_tokens", None),
                    latency_s=round(latency, 3),
                    attempts=attempt + 1,
                )
            except Exception as exc:
                latency = time.perf_counter() - t0
                last = f"{type(exc).__name__}: {exc}"
                # these will not fix themselves on retry
                if any(s in last.lower() for s in
                       ("invalid_api_key", "authentication", "not found",
                        "unsupported", "invalid_request")):
                    break
                time.sleep(min(2 ** attempt, 30))

        return CaptionResult(text=None, error=last,
                             latency_s=round(latency, 3), attempts=attempt + 1)

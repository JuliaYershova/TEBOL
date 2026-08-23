"""Prompt construction and the Captioner interface.
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

USER_PROMPT = "Describe this image in {n_words} words."


def build_prompt(n_words: int) -> str:
    return USER_PROMPT.format(n_words=n_words)


def max_tokens_for(n_words: int) -> int:
    """Generous budget so a caption is never truncated mid-sentence.

    The original notebook used a flat 300, which is plenty but wasteful; the
    packaged class used 20, which silently cut off anything past ~7 words.
    """
    return max(24, math.ceil(n_words * 4))


#: Gateways front their API with nginx, whose default client_max_body_size is
#: 1 MB. Measured against the gateway in use: a 1,077,948-byte base64 payload
#: is rejected with 413. The whole JSON request -- prompt, model name, the
#: data: URI header -- has to fit in that, so leave headroom below 1 MB.
MAX_B64_BYTES = 700_000


def encode_image(path: str | Path, max_b64_bytes: int = MAX_B64_BYTES) -> str:
    """Base64-encode an image, downscaling only if it would be rejected.

    Images under the limit -- 14,338 of 14,351 in this dataset -- are encoded
    byte-for-byte as they are on disk, so this changes nothing for almost every
    image. The handful of multi-megapixel outliers are shrunk until they fit
    rather than failing with a 413.
    """
    raw = Path(path).read_bytes()
    b64 = base64.b64encode(raw).decode("utf-8")
    if len(b64) <= max_b64_bytes:
        return b64

    from io import BytesIO

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            f"{Path(path).name} is {len(b64):,} base64 bytes, over the "
            f"{max_b64_bytes:,} limit, and Pillow is not installed to resize it "
            "-- pip install Pillow") from exc

    img = Image.open(BytesIO(raw))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    # shrink until it fits; 0.7 per step halves the pixel count each time
    for _ in range(8):
        img = img.resize((max(1, int(img.width * 0.7)),
                          max(1, int(img.height * 0.7))), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        if len(b64) <= max_b64_bytes:
            return b64
    return b64  # give the server the smallest we managed


@dataclass(frozen=True)
class CaptionResult:
    text: str | None
    error: str | None = None
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    #: seconds for the API call that produced this result (excludes retry waits)
    latency_s: float | None = None
    #: how many attempts were made, 1 when it worked first time
    attempts: int = 1


class Captioner(Protocol):
    """Anything that turns an image + word count into a caption."""

    name: str

    def caption(self, image_path: str | Path, n_words: int,
                temperature: float, seed: int | None = None) -> CaptionResult:
        ...

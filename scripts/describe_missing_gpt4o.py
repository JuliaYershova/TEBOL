#!/usr/bin/env python3
"""Describe the two images that have no stored GPT-4o description.

The original GPT-4o run (see build_gpt4o_inputs.py) left no description for two of
the 14,351 images, n07718472_4521 (cucumber) and n07590611_10775 (hot pot). This
script asks GPT-4o for them now, with the request the original notebook sent:

    one user message: the text "Describe this image in 7 words. Be concise, try to
    maximize the information about the image." and the image as a base64 data URL;
    max_tokens = 20; no system message, no temperature, no seed (API defaults).

One difference: the data URL carries the image's real MIME type, image/jpeg; the
original labelled the same JPEG bytes image/png.

These two descriptions are NOT part of the original run. They were generated later,
by whatever model snapshot `gpt-4o` pointed to at that time, which the reply
reports in `returned_model`. Everything the request and reply carry that matters
for judging this is written to data/gpt4o_added/descriptions.csv, one row per
image, and the file is read by build_gpt4o_inputs.py --added.

    python scripts/describe_missing_gpt4o.py --dry-run     # show the request, send nothing
    python scripts/describe_missing_gpt4o.py               # send two requests

The key is read from OPENAI_API_KEY or from .env and is never printed or stored. The
request goes through curl, as in embed_tokens_ada.py (see _post_embeddings there for
why).
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "gpt4o_added" / "descriptions.csv"

MISSING = {  # stem -> (pair, class)
    "n07718472_4521": ("cucumber_zucchini", "cucumber"),
    "n07590611_10775": ("hotpot_vase", "hotpot"),
}
MODEL = "gpt-4o"
PROMPT = ("Describe this image in 7 words. "
          "Be concise, try to maximize the information about the image.")
MAX_TOKENS = 20
FIELDS = ["stem", "pair", "class", "description", "requested_model", "returned_model",
          "prompt", "max_tokens", "temperature", "seed", "finish_reason",
          "prompt_tokens", "completion_tokens", "system_fingerprint", "generated_utc"]


def api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "OPENAI_API_KEY":
                return value.strip().strip("'\"")
    raise SystemExit("set OPENAI_API_KEY (environment or .env)")


def image_path(stem: str, pair: str, cls: str) -> Path:
    for ext in ("JPEG", "jpeg", "jpg", "JPG"):
        p = ROOT / "data" / "raw" / pair / cls / f"{stem}.{ext}"
        if p.exists():
            return p
    raise SystemExit(f"image of {stem} not found under data/raw/{pair}/{cls}")


def request_body(path: Path) -> dict:
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"model": MODEL, "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}]}


def post(body: dict, key: str) -> dict:
    """One request, one attempt: a failure is reported, not retried."""
    config = ('url = "https://api.openai.com/v1/chat/completions"\n'
              f'header = "Authorization: Bearer {key}"\n'
              'header = "Content-Type: application/json"\n')
    with tempfile.NamedTemporaryFile("w", suffix=".json") as fh:
        fh.write(json.dumps(body))
        fh.flush()
        out = subprocess.run(["curl", "-s", "-m", "120", "--config", "-",
                              "--data-binary", f"@{fh.name}"],
                             input=config, capture_output=True, text=True)
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        raise SystemExit(f"no valid reply: {out.stdout[:200] or out.stderr[:200]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the request without the image and send nothing")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    if args.out.exists():
        raise SystemExit(f"{args.out} exists; the two descriptions are generated once")

    rows = []
    for stem, (pair, cls) in MISSING.items():
        path = image_path(stem, pair, cls)
        body = request_body(path)
        if args.dry_run:
            shown = json.loads(json.dumps(body))
            shown["messages"][0]["content"][1]["image_url"]["url"] = (
                f"data:image/jpeg;base64,<{path.stat().st_size} bytes of {path.name}>")
            print(json.dumps(shown, indent=2))
            continue
        resp = post(body, api_key())
        if "choices" not in resp:
            raise SystemExit(f"{stem}: {resp.get('error', resp)}")
        choice = resp["choices"][0]
        usage = resp.get("usage", {})
        rows.append({
            "stem": stem, "pair": pair, "class": cls,
            "description": choice["message"]["content"],
            "requested_model": MODEL, "returned_model": resp.get("model"),
            "prompt": PROMPT, "max_tokens": MAX_TOKENS,
            "temperature": "", "seed": "",
            "finish_reason": choice.get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "system_fingerprint": resp.get("system_fingerprint"),
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
        print(f"{stem}: {rows[-1]['description']!r}  [{rows[-1]['returned_model']}]")

    if rows:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

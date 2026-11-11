#!/usr/bin/env python3
"""Embed a list of tokens with text-embedding-ada-002, one request per 256 tokens.

For a machine that may reach api.openai.com, when the machine that runs the
pipeline may not. Each token is embedded on its own, as the original GPT-4o run
did. The output has the format scripts/build_gpt4o_inputs.py reads as its cache,
so the pipeline machine only needs the file copied to

    artifacts/embeddings/openai__text-embedding-ada-002/_partial/cucumber_zucchini_api.npz

and `python scripts/build_gpt4o_inputs.py --embed-missing` then makes no API call.

    export OPENAI_API_KEY=...        # never written to a file
    python scripts/embed_tokens_ada.py logs/cucumber_zucchini_tokens_needed.txt cucumber_zucchini_api.npz
"""
import os
import sys

import numpy as np
from openai import OpenAI

src, dst = sys.argv[1], sys.argv[2]
tokens = [t for t in open(src, encoding="utf-8").read().split("\n") if t]
client = OpenAI(max_retries=3, timeout=120)
keys, vectors = [], []
for i in range(0, len(tokens), 256):
    chunk = tokens[i:i + 256]
    resp = client.embeddings.create(model="text-embedding-ada-002", input=chunk)
    for tok, item in zip(chunk, sorted(resp.data, key=lambda d: d.index)):
        keys.append(tok)
        vectors.append(np.asarray(item.embedding, dtype=np.float64))
    print(f"{min(i + 256, len(tokens))}/{len(tokens)}", flush=True)
np.savez_compressed(dst, keys=np.asarray(keys, dtype=object), vectors=np.stack(vectors))
print("wrote", dst)

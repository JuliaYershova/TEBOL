"""Caption tokenisation.

Every downstream stage keys on words: embeddings are computed per word, SMER
scores per word, LIME perturbs words, AOPC removes words. So the split from
caption string to token list has to happen exactly once, here, or the same word
ends up as several different features.

Two failure modes this exists to prevent, both visible in the raw Qwen output:

    'vase.' vs 'vase'        -- 23% of tokens carry trailing punctuation, and
                                string equality makes them separate features
    'Blue-and-white'         -- one token or three? Either is defensible, but
                                it must be decided once, not per notebook.
"""

from __future__ import annotations

import re
import unicodedata

# stripped from the edges of a token, never from the middle
EDGE_PUNCT = ".,;:!?\"'`()[]{}<>…«»„“”‘’*_/\\|"

# curly punctuation -> ascii, so 'don’t' and "don't" are one token
_UNIFY = {"‘": "'", "’": "'", "“": '"', "”": '"',
          "–": "-", "—": "-", "−": "-", " ": " "}

_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """NFKC, unify punctuation, collapse whitespace, lowercase."""
    text = unicodedata.normalize("NFKC", text or "")
    for a, b in _UNIFY.items():
        text = text.replace(a, b)
    return _WS.sub(" ", text).strip().lower()


def normalize_token(tok: str) -> str:
    """Strip edge punctuation but keep internal hyphens and apostrophes."""
    return tok.strip(EDGE_PUNCT)


def tokenize(caption: str, split_hyphens: bool = False) -> list[str]:
    """Caption -> word list.

    split_hyphens=False keeps 'red-figure' as one token, which matches how the
    Diplom pipeline split on whitespace. Set True to treat it as 'red',
    'figure' -- more tokens, but colour/texture compounds get separate
    importance scores.
    """
    tokens = []
    for raw in normalize_text(caption).split():
        tok = normalize_token(raw)
        if not tok:
            continue
        if split_hyphens and "-" in tok:
            tokens.extend(p for p in (normalize_token(p) for p in tok.split("-")) if p)
        else:
            tokens.append(tok)
    return tokens


def normalize_caption(caption: str, split_hyphens: bool = False) -> str:
    """Tokenise and rejoin -- the canonical string downstream stages consume."""
    return " ".join(tokenize(caption, split_hyphens))

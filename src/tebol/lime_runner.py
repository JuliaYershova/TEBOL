"""One caption's LIME attribution, isolated so it can run in a worker process.

This lives in the package rather than in scripts/09_lime.py because joblib's
loky backend pickles a task function by module path. A function defined in a
`__main__` script fails to un-serialize in the worker, and the failure arrives
as a bare BrokenProcessPool with no indication of the cause.

The classifier LIME is asked about is not a model object -- it is the closed
form stage 3 already reduced the model to:

    p = expit(bias + mean(z[kept]))

so a worker needs only a vector of per-word scalars, not the 2560-dimensional
vectors, the coefficients, or anything that would have to be shipped to it.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit

#: LIME marks a removed position with this sentinel when bow=False, rather than
#: deleting it, so the perturbation mask is recoverable from the string.
SENTINEL = "UNKWORDZ"


def explain_one(words: list[str], z: np.ndarray, bias: float,
                toward_pos: bool, num_samples: int, seed: int,
                legacy: bool = False, bow: bool = False) -> np.ndarray:
    """LIME weight per position, in position order.

    `toward_pos` selects which class is explained: always the one the model
    predicted, so a positive weight means "supports the prediction" -- the
    convention SMER uses and the one AOPC needs.
    """
    from lime.lime_text import LimeTextExplainer

    n = len(words)
    if n < 2:
        return np.zeros(n)

    def predict(texts):
        out = np.empty((len(texts), 2))
        for j, t in enumerate(texts):
            toks = t.split()
            if legacy or bow:
                # bow=True deletes every occurrence of a word, so the mask is
                # recovered by membership rather than by position.
                present = set(toks)
                kept = [z[i] for i, w in enumerate(words) if w in present]
            else:
                kept = [z[i] for i, tok in enumerate(toks)
                        if i < n and tok != SENTINEL]
            p1 = expit(bias + (float(np.mean(kept)) if kept else 0.0))
            out[j] = (1.0 - p1, p1)
        return out

    if legacy:
        # The Diplom notebook's exact configuration, kept so the published
        # comparison can be reproduced and contrasted: bow=True, the \W+
        # tokeniser, num_features=15, the default labels=(1,) -- so class index
        # 1 is explained whatever the model predicted -- and abs() applied to
        # the weights. The abs() is the consequential one: it discards
        # direction, so a word that pushes *away* from the prediction scores as
        # highly as one that supports it, and AOPC then removes words that can
        # raise the predicted probability rather than lower it.
        expl = LimeTextExplainer(random_state=seed)
        exp = expl.explain_instance(" ".join(words), predict, labels=(1,),
                                    num_features=15, num_samples=num_samples)
        w = np.zeros(n)
        m = dict(exp.as_map()[1])
        for i in range(n):
            w[i] = abs(m.get(i, 0.0))
        return w

    # bow=False makes each position its own feature; bow=True is LIME's own
    # default and removes every occurrence of a word together, which loses the
    # distinction between two copies of the same word. Both are defensible and
    # both are run; the weights stay signed and the predicted class is the one
    # explained either way, so only the tokenisation differs.
    expl = LimeTextExplainer(bow=bow, random_state=seed,
                             split_expression=r"\W+" if bow else r"\s+")
    label = 1 if toward_pos else 0
    exp = expl.explain_instance(" ".join(words), predict, labels=(label,),
                                num_features=n, num_samples=num_samples)

    w = np.zeros(n)
    for pos, weight in exp.as_map()[label]:
        if 0 <= pos < n:
            w[pos] = weight
    return w

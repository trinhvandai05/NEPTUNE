"""Head/tail partition from TRAIN interactions only (amendment A1.3).

The function signature takes train items and nothing else; the covariate
builder passes it split==TRAIN events only.  tests/test_train_only_popularity.py
injects huge future popularity into test and checks head membership is unchanged.
"""

from __future__ import annotations

import numpy as np


def head_mask_from_train(train_items: np.ndarray, n_items: int, fraction: float) -> np.ndarray:
    counts = np.bincount(np.asarray(train_items, dtype=np.int64), minlength=n_items)
    n_head = int(np.ceil(fraction * n_items))
    order = np.lexsort((np.arange(n_items), -counts))   # count desc, index asc
    head = np.zeros(n_items, dtype=bool)
    head[order[:n_head]] = True
    return head & (counts > 0)                           # never-seen items are never head


def tail_fraction(items: np.ndarray, head: np.ndarray) -> float:
    items = np.asarray(items, dtype=np.int64)
    if len(items) == 0:
        return float("nan")
    return float((~head[items]).mean())

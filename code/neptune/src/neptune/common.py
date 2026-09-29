"""Small shared utilities.

This module must stay free of any model import: the semantics layer (which
defines the H1 stratification variable) is allowed to import it, and the
semantics layer is forbidden from seeing the model.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def write_json_atomic(path, obj) -> None:
    """Write-to-temp then os.replace: a crash mid-write never leaves a
    truncated file that looks valid."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_json_default))
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def quantile_labels(x, q: int) -> np.ndarray:
    """Balanced labels 0..q-1 by rank.

    Rank-based rather than value-based edges, so heavily tied variables (history
    length is integer-valued) never produce empty or merged bins.  Ties are
    broken by position, which is deterministic given a fixed row order.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(n)
    return np.minimum((ranks * q) // n, q - 1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass

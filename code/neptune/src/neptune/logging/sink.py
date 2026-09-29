"""Long-format metric sink.

One row per measurement with a fixed schema

    experiment, arm, dataset, seed, epoch, split, user_id, user_stratum, metric, value

so every diagnostic is an offline query over one table.  Rows stream to JSONL
(crash-safe, append-only) and are compacted to Parquet on close.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

FIELDS = ("experiment", "arm", "dataset", "seed", "epoch", "split",
          "user_id", "user_stratum", "metric", "value")


class LongSink:
    def __init__(self, path, base: dict):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.base = {k: base.get(k) for k in FIELDS if k in base}
        self.f = open(self.path, "a", buffering=1 << 16)

    def log(self, metric: str, value, **fields):
        row = {k: None for k in FIELDS}
        row.update(self.base)
        row.update(fields)
        row["metric"] = metric
        row["value"] = None if value is None else float(value)
        self.f.write(json.dumps(row) + "\n")

    def close(self):
        if self.f.closed:
            return
        self.f.close()
        if self.path.stat().st_size:
            df = pd.read_json(self.path, lines=True)
            df.to_parquet(self.path.with_suffix(".parquet"), index=False)

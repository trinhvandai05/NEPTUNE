"""Length-bucketed batches of whole user sequences.

Throughput comes entirely from processing many users at the same event index,
so users are bucketed by length (a single 2000-event user would otherwise force
its whole batch to 2000 steps).  Shapes are fixed-size tensors; no per-user
Python objects cross into the model.

mode="train": only TRAIN events (all of them when max_len == 0, the preregistered value).
mode="eval" : the full sequence (train + val + test), so the state is rolled
              through history before each evaluated event.
Batch order per epoch is a pure function of (seed, epoch), which is what makes
mid-epoch resume exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

DAY = 86400.0


@dataclass
class Sequences:
    items: list
    times: list
    split: list
    users: np.ndarray


def build_sequences(events: pd.DataFrame, n_users: int) -> Sequences:
    ev = events.sort_values(["u", "t"], kind="mergesort")
    u = ev["u"].to_numpy()
    bounds = np.flatnonzero(np.r_[True, u[1:] != u[:-1], True])
    it, tm, sp = ev["i"].to_numpy(), ev["t"].to_numpy(), ev["split"].to_numpy()
    items, times, split, users = [], [], [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        items.append(it[a:b].astype(np.int64))
        times.append(tm[a:b].astype(np.float64))
        split.append(sp[a:b].astype(np.int8))
        users.append(int(u[a]))
    return Sequences(items, times, split, np.asarray(users, dtype=np.int64))


def deltas(times: np.ndarray, ordinal: bool, clip_days: float) -> np.ndarray:
    if ordinal:
        return np.ones(len(times), dtype=np.float32)
    dt = np.diff(times, prepend=times[0]) / DAY
    dt = np.log1p(np.clip(dt, 0.0, clip_days)).astype(np.float32)
    return np.maximum(dt, 1e-3)


class BucketedBatcher:
    def __init__(self, seqs: Sequences, batch_size: int, *, mode: str, ordinal: bool,
                 clip_days: float = 365.0, max_len: int | None = None, seed: int = 0):
        if mode not in ("train", "eval"):
            raise ValueError(mode)
        self.mode, self.bs, self.seed = mode, batch_size, seed
        rows = []
        for k in range(len(seqs.items)):
            it, tm, sp = seqs.items[k], seqs.times[k], seqs.split[k]
            if mode == "train":
                keep = sp == 0
                it, tm, sp = it[keep], tm[keep], sp[keep]
                if max_len and len(it) > max_len:          # 0 / None = keep full history
                    it, tm, sp = it[-max_len:], tm[-max_len:], sp[-max_len:]
                if len(it) < 2:
                    continue
            rows.append((seqs.users[k], it, deltas(tm, ordinal, clip_days), sp))
        rows.sort(key=lambda r: len(r[1]))
        self.groups = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]

    def __len__(self):
        return len(self.groups)

    def order(self, epoch: int) -> np.ndarray:
        idx = np.arange(len(self.groups))
        if self.mode == "train":
            np.random.default_rng((self.seed, epoch)).shuffle(idx)
        return idx

    def steps_per_epoch(self, bptt: int) -> int:
        return int(sum(int(np.ceil(max(len(r[1]) for r in g) / bptt)) for g in self.groups))

    def n_events(self) -> int:
        return int(sum(len(r[1]) for g in self.groups for r in g))

    def batch(self, g: int) -> dict:
        grp = self.groups[g]
        B, L = len(grp), max(len(r[1]) for r in grp)
        items = np.zeros((B, L), np.int64)
        dt = np.zeros((B, L), np.float32)
        split = np.full((B, L), -1, np.int8)
        mask = np.zeros((B, L), bool)
        for b, (_, it, d, sp) in enumerate(grp):
            n = len(it)
            items[b, :n], dt[b, :n], split[b, :n], mask[b, :n] = it, d, sp, True
        return {"items": torch.from_numpy(items), "dt": torch.from_numpy(dt),
                "split": torch.from_numpy(split), "mask": torch.from_numpy(mask),
                "users": torch.as_tensor([r[0] for r in grp], dtype=torch.long)}

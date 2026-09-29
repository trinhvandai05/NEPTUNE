"""Crash-resilient checkpointing.

Stores model, optimizer, scheduler, every RNG stream (torch, cuda, numpy,
python, the device negative-sampler generator) and the batch position, so a
resumed run continues bit-identically from that point.  Writes are atomic
(tmp + os.replace): a checkpoint truncated by a crash mid-write is worse than
none, because it looks valid.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def save(path, *, model, optimizer, scheduler, sampler, epoch: int, batch_pos: int,
         step: int, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "sampler": sampler.state_dict() if sampler is not None else None,
        "epoch": epoch, "batch_pos": batch_pos, "step": step,
        "rng": {"torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(), "python": random.getstate()},
        "extra": extra or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load(path, *, model, optimizer=None, scheduler=None, sampler=None, map_location=None) -> dict:
    # Always load to CPU: RNG states must be CPU ByteTensors (map_location="cuda"
    # would move them and set_rng_state would reject them).  load_state_dict copies
    # into the live parameters, and the optimizer casts its state to their device.
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ck["optimizer"])
    if scheduler is not None and ck.get("scheduler") is not None:
        scheduler.load_state_dict(ck["scheduler"])
    if sampler is not None and ck.get("sampler") is not None:
        sampler.load_state_dict(ck["sampler"])
    rng = ck["rng"]
    torch.set_rng_state(rng["torch"])
    if rng.get("cuda") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(rng["cuda"])
        except Exception as e:  # different device count on resume
            print(f"[checkpoint] cuda RNG not restored: {e}")
    np.random.set_state(rng["numpy"])
    random.setstate(rng["python"])
    return ck

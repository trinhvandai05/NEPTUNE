"""Negative samplers.

Both samplers exist and BOTH must be reported: the sampler is itself a
popularity-bias channel, so which one appears in the paper cannot be chosen
after seeing results.  Negatives are drawn only from WARM items (seen in train):
a cold item used as a negative would hand the model gradient on exactly the
items the cold-start stratum claims it has never trained on.

Sampling happens on the device with a device generator: no per-step
host-to-device copy, and the generator state is checkpointed.
"""

from __future__ import annotations

import numpy as np
import torch


class NegativeSampler:
    def __init__(self, allowed: np.ndarray, popularity: np.ndarray, mode: str, device, seed: int):
        if mode not in ("uniform", "popularity"):
            raise ValueError(f"unknown negative sampler '{mode}'")
        idx = np.flatnonzero(allowed)
        if len(idx) == 0:
            raise ValueError("no warm items to sample negatives from")
        self.mode = mode
        self.device = torch.device(device)
        self.items = torch.as_tensor(idx, dtype=torch.long, device=self.device)
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(int(seed))
        self.probs = None
        if mode == "popularity":
            w = np.power(np.maximum(popularity[idx].astype(np.float64), 1.0), 0.75)
            self.probs = torch.as_tensor(w / w.sum(), dtype=torch.float32, device=self.device)

    def sample(self, B: int, S: int) -> torch.Tensor:
        if self.mode == "uniform":
            k = torch.randint(0, len(self.items), (B, S), generator=self.gen, device=self.device)
        else:
            k = torch.multinomial(self.probs, B * S, replacement=True, generator=self.gen).view(B, S)
        return self.items[k]

    def state_dict(self):
        return {"gen": self.gen.get_state()}

    def load_state_dict(self, sd):
        self.gen.set_state(sd["gen"].cpu())

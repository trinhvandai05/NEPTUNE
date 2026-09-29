"""e_phi : R^{d_x} -> S^{d-1}.  No item-ID embedding exists anywhere in NEPTUNE.

That is not an efficiency choice: it is what makes the cold-item stratum
meaningful.  An item with zero training interactions still has coordinates on
the sphere, so it receives preference density on step one.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..state.kernel import renormalize


class ItemEncoder(nn.Module):
    def __init__(self, d_x: int, d: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_x, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return renormalize(self.net(x))

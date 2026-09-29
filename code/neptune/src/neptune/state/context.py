"""Event assimilation channel (spec sec. 13.1, gap G1).

    c = [ e_{last}, ema, phi_time(dt) ]

Without this the flow is autonomous and nothing ever pulls the measure toward
a newly consumed item (the satiation deposit is repulsive).  In filtering
terms the flow is the prediction step and assimilation is the update step.

The two operations are separate methods on purpose:
  * context_for_interval() reads the PREVIOUS observation -- used to predict i_t
  * assimilate()           folds i_t in -- only after the heads have run
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ContextEncoder(nn.Module):
    def __init__(self, d: int, n_time_feats: int = 8, ema_beta: float = 0.2):
        super().__init__()
        self.ema_beta = float(ema_beta)
        self.out_dim = 2 * d + n_time_feats
        self.register_buffer("freqs", torch.exp(torch.linspace(-3.0, 3.0, n_time_feats)))

    def time_features(self, dt: torch.Tensor) -> torch.Tensor:
        return torch.tanh(torch.log1p(dt.clamp_min(0.0)).unsqueeze(-1) * self.freqs)

    def context_for_interval(self, last_emb: torch.Tensor, ema: torch.Tensor,
                             dt: torch.Tensor) -> torch.Tensor:
        return torch.cat([last_emb, ema, self.time_features(dt)], dim=-1)

    def assimilate(self, ema: torch.Tensor, e_item: torch.Tensor,
                   active: torch.Tensor) -> torch.Tensor:
        new = (1.0 - self.ema_beta) * ema + self.ema_beta * e_item
        return torch.where(active.unsqueeze(-1), new, ema)

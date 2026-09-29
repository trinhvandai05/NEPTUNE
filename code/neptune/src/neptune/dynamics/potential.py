"""V_theta(z; c) = - sum_j w_j(c) kappa_{h_V}(z, u_j(c)).

DELIBERATE DEVIATION from spec sec. 5.1 (3-layer MLP evaluated at z):
  1. grad_z V is closed-form.  A black-box MLP needs
     autograd.grad(create_graph=True) inside every RK4 stage -- a
     double-backward per stage, expensive and memory-hungry on 24 GB.
  2. It makes the G1 assimilation channel inspectable: attractors are placed
     from c_t, which contains e_{i_{t-1}}, so the just-consumed item visibly
     pulls the landscape toward itself.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..state.kernel import kernel, kernel_ambient_grad, renormalize


class AttractorPotential(nn.Module):
    def __init__(self, ctx_dim: int, d: int, n_attractors: int = 8,
                 hidden: int = 256, h_v: float = 0.45):
        super().__init__()
        self.P, self.d, self.h_v = n_attractors, d, h_v
        self.net = nn.Sequential(nn.Linear(ctx_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, n_attractors * (d + 1)))

    def emit(self, ctx: torch.Tensor):
        out = self.net(ctx).view(ctx.shape[0], self.P, self.d + 1)
        return renormalize(out[..., : self.d]), F.softplus(out[..., self.d])

    def value(self, z, u, w):
        return -(kernel(z, u, self.h_v) * w.unsqueeze(1)).sum(-1)

    def ambient_grad(self, z, u, w):
        return -kernel_ambient_grad(z, u, w, self.h_v)

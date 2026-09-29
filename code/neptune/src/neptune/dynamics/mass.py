"""Normalized replicator on the mass simplex.

    p' = softmax( log p - alpha * eta * dt * f )

log_softmax form, never exponentiate-then-renormalize: it lives on the simplex
by construction.  (The unnormalized version inflates total mass by
~ (c^2/2) Var_mu[f] per step by Jensen -- a pure discretization artifact.)
Invariant to adding a constant to log p or to f: the gauge test checks this.
"""

from __future__ import annotations

import torch


def replicator_step(logp: torch.Tensor, f: torch.Tensor, alpha: float,
                    eta: float, dt: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(logp - alpha * eta * dt.unsqueeze(-1) * f, dim=-1)

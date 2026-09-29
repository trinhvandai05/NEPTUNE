"""Fixed-step RK4 -- the production solver for H1.

Stability cap  h_j <= h_max = delta_safe / (eta * r_max)  forces
J_t = ceil(Delta_t / h_max) = Omega(Delta_t); no step-spacing schedule evades
that (an accuracy heuristic cannot relax a stability floor).  With the H1
ordinal profile Delta_t = 1 < h_max = 2.5, so J = 1: one vectorized RK4 step
per event.

Adaptive Dormand-Prince is NOT part of this slice.  Per-user adaptive step
counts break batching, which is worth ~100x on this workload; it returns later
only as diagnostic D8.  J is computed on the CPU from known gaps, so choosing
it never forces a device sync.
"""

from __future__ import annotations

import math

import torch

from ..state.kernel import renormalize


def n_substeps(dt_max: float, h_max: float) -> int:
    return max(1, int(math.ceil(float(dt_max) / h_max - 1e-12)))


def rk4_interval(field, z: torch.Tensor, p: torch.Tensor, t0: torch.Tensor,
                 dt: torch.Tensor, eta: float, J: int) -> torch.Tensor:
    """Every user takes J steps of size dt_u / J (never more than h_max, never
    padded).  Re-normalized onto the sphere after every full step."""
    h = (dt / J).view(-1, 1, 1) * eta
    th = dt / J
    now = t0
    for _ in range(J):
        k1 = field(z, p, now)
        k2 = field(renormalize(z + 0.5 * h * k1), p, now + 0.5 * th)
        k3 = field(renormalize(z + 0.5 * h * k2), p, now + 0.5 * th)
        k4 = field(renormalize(z + h * k3), p, now + th)
        z = renormalize(z + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))
        now = now + th
    return z

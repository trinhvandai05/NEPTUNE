"""Sphere primitives and the Gaussian-on-sphere kernel.

On the unit sphere ||x - y||^2 = 2 - 2<x,y>, so

    kappa_h(x, y) = exp( (<x,y> - 1) / h^2 )

is exactly a Gaussian kernel of bandwidth h.  The shifted form keeps the
exponent <= 0 so exp() cannot overflow.  All kernel math runs in fp32.
"""

from __future__ import annotations

import torch

LOG_CLAMP = -60.0


def renormalize(z: torch.Tensor) -> torch.Tensor:
    return z / z.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def project(z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """P_z v = v - <z,v> z : ambient vector -> tangent space at z."""
    return v - (z * v).sum(-1, keepdim=True) * z


def cosine(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x: [..., A, d], y: [..., B, d] -> [..., A, B].

    Clamped to [-1, 1]: after renormalize, <z,z> can land at 1 + 1e-7 in fp32,
    which would push kappa above its bound of 1 and break log-kernel <= 0.
    """
    return torch.einsum("...ad,...bd->...ab", x, y).clamp(-1.0, 1.0)


def log_kernel(x: torch.Tensor, y: torch.Tensor, h: float) -> torch.Tensor:
    return (cosine(x, y) - 1.0) / (h * h)


def kernel(x: torch.Tensor, y: torch.Tensor, h: float) -> torch.Tensor:
    return torch.exp(log_kernel(x, y, h).clamp_min(LOG_CLAMP))


def kernel_ambient_grad(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, h: float) -> torch.Tensor:
    """Ambient d/dx of sum_b w_b kappa_h(x_a, y_b).  x [...,A,d], y [...,B,d], w [...,B].

    d kappa/dx = kappa * y / h^2.  Returned UNPROJECTED; the caller projects the
    total once.
    """
    kw = kernel(x, y, h) * w.unsqueeze(-2)
    return torch.einsum("...ab,...bd->...ad", kw, y) / (h * h)

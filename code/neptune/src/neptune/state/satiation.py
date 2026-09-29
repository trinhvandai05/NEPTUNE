"""Semantic satiation field Psi and the atom bank that carries it.

    Psi_u^t(z) = sum_r a(i_r) exp(-lambda(i_r) (t - t_r)) kappa_sigma(z, e_{i_r})

Atoms store ITEM INDICES, not embeddings: the encoder trains, so an atom's
position must be re-gathered from the current encoder at use time.

The bank is shared verbatim by NEPTUNE and the SASRec+Psi control, so the
control really does get "the same Psi".
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kernel import kernel


@dataclass
class AtomBank:
    item: torch.Tensor    # [B, R] long
    amp: torch.Tensor     # [B, R] a_psi at deposit
    lam: torch.Tensor     # [B, R] lambda_phi
    t: torch.Tensor       # [B, R] deposit time (model time units)
    mask: torch.Tensor    # [B, R] bool

    @staticmethod
    def empty(B: int, R: int, device, dtype=torch.float32) -> "AtomBank":
        z = torch.zeros(B, R, dtype=dtype, device=device)
        return AtomBank(item=torch.zeros(B, R, dtype=torch.long, device=device),
                        amp=z, lam=torch.ones_like(z), t=z.clone(),
                        mask=torch.zeros(B, R, dtype=torch.bool, device=device))

    def detach(self) -> "AtomBank":
        return AtomBank(self.item, self.amp.detach(), self.lam.detach(), self.t.detach(), self.mask)

    def index(self, idx) -> "AtomBank":
        return AtomBank(self.item[idx], self.amp[idx], self.lam[idx], self.t[idx], self.mask[idx])

    def tensors(self):
        return (self.item, self.amp, self.lam, self.t, self.mask)


class SatiationParams(nn.Module):
    """Two learned scalars per item from RAW content: amplitude and recovery rate.
    The entire depletion-regeneration law of the catalog lives in these heads."""

    def __init__(self, d_x: int, hidden: int = 128):
        super().__init__()
        self.amp = nn.Sequential(nn.Linear(d_x, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.lam = nn.Sequential(nn.Linear(d_x, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor):
        a = F.softplus(self.amp(x).squeeze(-1)).clamp_min(1e-4)
        lam = F.softplus(self.lam(x).squeeze(-1)).clamp_min(1e-4)
        return a, lam


def contributions(bank: AtomBank, now: torch.Tensor) -> torch.Tensor:
    """Current weight of each atom, a * exp(-lambda * age), zero for dead slots. [B, R]"""
    age = (now.unsqueeze(-1) - bank.t).clamp_min(0.0)
    return bank.amp * torch.exp(-bank.lam * age) * bank.mask.to(bank.amp.dtype)


def psi_from_contrib(points: torch.Tensor, e_atom: torch.Tensor, contrib: torch.Tensor,
                     sigma: float, want_grad: bool = False):
    """points [B, Q, d], e_atom [B, R, d], contrib [B, R] -> Psi [B, Q] (+ ambient grad)."""
    kc = kernel(points, e_atom, sigma) * contrib.unsqueeze(1)       # [B, Q, R]
    psi = kc.sum(-1)
    if not want_grad:
        return psi, None
    return psi, torch.einsum("bqr,brd->bqd", kc, e_atom) / (sigma * sigma)


def prune(bank: AtomBank, now: torch.Tensor, active: torch.Tensor, eps: float) -> AtomBank:
    """Step 1 of the event order (spec sec. 5.5): decay to tau_t and drop atoms
    whose contribution fell below eps -- BEFORE the flow and the heads see Psi.
    Time-driven only; it never looks at the current item."""
    dead = bank.mask & (contributions(bank, now) < eps) & active.unsqueeze(1)
    return AtomBank(bank.item, bank.amp, bank.lam, bank.t, bank.mask & ~dead)


def deposit(bank: AtomBank, item: torch.Tensor, amp: torch.Tensor, lam: torch.Tensor,
            now: torch.Tensor, active: torch.Tensor, eps: float):
    """Write the new atom (the eps check here is idempotent after prune()).

    Slot choice: a dead slot if any exists, otherwise the weakest live atom.
    Returns (bank, evicted_live[B]) -- evicted_live marks capacity truncation
    (a still-relevant atom thrown out because R is full).  A high rate means R
    is too small and Psi is being silently truncated (diagnostic D12).
    """
    R = bank.item.shape[1]
    cur = contributions(bank, now)
    alive = bank.mask & (cur >= eps)
    key = torch.where(alive, cur, torch.full_like(cur, -1.0))
    slot = key.argmin(dim=-1)
    evicted = alive.gather(1, slot.unsqueeze(1)).squeeze(1) & active
    write = F.one_hot(slot, R).bool() & active.unsqueeze(1)
    new = AtomBank(
        item=torch.where(write, item.unsqueeze(1), bank.item),
        amp=torch.where(write, amp.unsqueeze(1), bank.amp),
        lam=torch.where(write, lam.unsqueeze(1), bank.lam),
        t=torch.where(write, now.unsqueeze(1), bank.t),
        mask=torch.where(active.unsqueeze(1), alive | write, bank.mask),
    )
    return new, evicted

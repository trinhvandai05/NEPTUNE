"""Ranking head.

    s(i) = log mu_bar(e_i) - Psi(e_i)
         = logsumexp_k [ log p_k + (<e_i, z_k> - 1)/h^2 ] - Psi(e_i)

Evaluated in log space; the density is never materialized.  Full-catalog
ranking is chunked over items so [B, N, M] is never allocated.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..state.kernel import log_kernel
from ..state.satiation import contributions, psi_from_contrib


def log_density(z: torch.Tensor, logp: torch.Tensor, E_query: torch.Tensor, h: float) -> torch.Tensor:
    return torch.logsumexp(log_kernel(E_query, z, h) + logp.unsqueeze(1), dim=-1)


def score(state, E_query: torch.Tensor, E_all: torch.Tensor, *, h: float, sigma: float,
          use_satiation: bool) -> torch.Tensor:
    """E_query: [B, Q, d] per-user, or [Q, d] shared across the batch."""
    B = state.z.shape[0]
    if E_query.dim() == 2:
        E_query = E_query.unsqueeze(0).expand(B, -1, -1)
    s = log_density(state.z, state.logp, E_query, h)
    if use_satiation:
        e_atom = E_all[state.atoms.item]
        psi, _ = psi_from_contrib(E_query, e_atom, contributions(state.atoms, state.time), sigma)
        s = s - psi
    return s


def sampled_softmax_ce(scores: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    """Cross-entropy with the positive in column 0.  Accidental hits (a negative
    equal to the positive) are masked out rather than counted as negatives."""
    hit = neg == pos.unsqueeze(1)
    scores = torch.cat([scores[:, :1], scores[:, 1:].masked_fill(hit, float("-inf"))], dim=1)
    target = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)
    return F.cross_entropy(scores, target, reduction="none")


@torch.no_grad()
def rank_against_catalog(score_chunk, target_idx: torch.Tensor, target_score: torch.Tensor,
                         seen: torch.Tensor, n_items: int, chunk: int) -> torch.Tensor:
    """1 + number of UNSEEN, NON-TARGET items scoring strictly above the target.

    score_chunk(lo, hi) -> [b, hi - lo].  The target column is excluded by INDEX,
    not by relying on `>`: the target's score recomputed inside a chunk of a
    different shape can differ from `target_score` by one ulp (einsum picks
    different kernels per shape), which would let the target "beat itself" and
    make the rank depend on the chunk size.  tests/test_full_catalog.py pins this.
    """
    greater = torch.zeros_like(target_score, dtype=torch.long)
    for lo in range(0, n_items, chunk):
        hi = min(n_items, lo + chunk)
        s = score_chunk(lo, hi)
        cols = torch.arange(lo, hi, device=s.device)
        is_target = cols.unsqueeze(0) == target_idx.unsqueeze(1)
        greater += ((s > target_score.unsqueeze(1)) & ~seen[:, lo:hi] & ~is_target).sum(dim=1)
    return greater + 1

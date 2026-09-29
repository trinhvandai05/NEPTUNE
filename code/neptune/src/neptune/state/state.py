"""H1 state.

    S_u^t = ( mu_bar = sum_k p_k delta_{z_k},  Psi )

There is NO log Z here.  With omega = 0 the ranking loss is exactly
scale-invariant in the particle masses, so the radial coordinate would receive
no gradient at all; it is post-H1 machinery and does not exist in this slice.
The interaction coefficient gamma * sg[Z] therefore reduces to gamma.

Observation fields (atoms, ema, last_item, has_last) may only change inside
NeptuneH1.assimilate().  event_step() verifies this by tensor identity and
version counter -- no device sync.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .kernel import renormalize
from .satiation import AtomBank


@dataclass
class H1State:
    z: torch.Tensor          # [B, M, d]
    logp: torch.Tensor       # [B, M]  log masses, log_softmax-normalized
    atoms: AtomBank
    ema: torch.Tensor        # [B, d]
    last_item: torch.Tensor  # [B] long -- re-gathered from the current encoder
    has_last: torch.Tensor   # [B] bool
    time: torch.Tensor       # [B]

    def observation_tensors(self):
        return (*self.atoms.tensors(), self.ema, self.last_item, self.has_last)

    def detach(self) -> "H1State":
        return H1State(self.z.detach(), self.logp.detach(), self.atoms.detach(),
                       self.ema.detach(), self.last_item, self.has_last, self.time.detach())

    def index(self, idx) -> "H1State":
        return H1State(self.z[idx], self.logp[idx], self.atoms.index(idx), self.ema[idx],
                       self.last_item[idx], self.has_last[idx], self.time[idx])

    @property
    def p(self) -> torch.Tensor:
        return self.logp.exp()

    @property
    def ess(self) -> torch.Tensor:
        return 1.0 / self.p.pow(2).sum(-1).clamp_min(1e-12)


def init_state(B: int, prior_z: torch.Tensor, prior_logp: torch.Tensor, R: int, device) -> H1State:
    M, d = prior_z.shape
    return H1State(
        z=renormalize(prior_z).unsqueeze(0).expand(B, M, d),
        logp=torch.log_softmax(prior_logp, -1).unsqueeze(0).expand(B, M),
        atoms=AtomBank.empty(B, R, device, prior_z.dtype),
        ema=torch.zeros(B, d, device=device, dtype=prior_z.dtype),
        last_item=torch.zeros(B, dtype=torch.long, device=device),
        has_last=torch.zeros(B, dtype=torch.bool, device=device),
        time=torch.zeros(B, device=device, dtype=prior_z.dtype),
    )

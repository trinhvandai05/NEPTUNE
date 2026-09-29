"""NEPTUNE-H1: the smallest model that can falsify claim C-I.

What is IN this slice
  item encoder e_phi (no IDs), particles (z, log p), satiation atoms Psi,
  event-conditioned attractor potential, angular flow, replicator, RK4.

What is deliberately OUT (post-H1, unlocked only if H1 survives)
  log Z / OU intensity, rho(c), time likelihood, compensator, RFF catalog
  moment, HNSW, budget residual, drag estimator, power-iteration growth rate.

Prior particles are initialized at the encoder images of M random catalog
items -- NOT at the semantic k-means centroids.  Initializing at the centroids
would create an information path from the H1 stratification variable into the
model, which is precisely the circularity the semantics isolation rule forbids.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import RunConfig
from .content.encoder import ItemEncoder
from .dynamics.angular import AngularField
from .dynamics.mass import replicator_step
from .dynamics.potential import AttractorPotential
from .dynamics.rk4 import n_substeps, rk4_interval
from .heads import ranking
from .state.context import ContextEncoder
from .state.satiation import SatiationParams, deposit, prune
from .state.state import H1State, init_state


class NeptuneH1(nn.Module):
    def __init__(self, cfg: RunConfig, X_raw: torch.Tensor, seed: int):
        super().__init__()
        m, s = cfg.model, cfg.solver
        d_x = X_raw.shape[1]
        self.M, self.d, self.R = m.M, m.d, m.R
        self.h, self.sigma, self.gamma = m.h, m.sigma, m.gamma
        self.alpha, self.eta, self.atom_eps = m.alpha, m.eta, m.atom_eps
        self.use_satiation = m.satiation_enabled
        self.lambda_dis = cfg.reg.lambda_dis
        self.ess_floor = cfg.reg.ess_min_frac * m.M
        self.h_max = s.delta_safe / (m.eta * s.r_max)

        self.encoder = ItemEncoder(d_x, m.d, m.encoder_hidden)
        self.context = ContextEncoder(m.d, m.n_time_feats, m.ema_beta)
        self.potential = AttractorPotential(self.context.out_dim, m.d, m.n_attractors,
                                            m.potential_hidden, m.h_v)
        self.satiation = SatiationParams(d_x, m.satiation_hidden)

        g = torch.Generator().manual_seed(int(seed))
        n = X_raw.shape[0]
        idx = torch.randperm(n, generator=g)[: m.M] if n >= m.M else torch.randint(0, n, (m.M,), generator=g)
        with torch.no_grad():
            # the encoder is still on CPU here (the caller moves the model later),
            # so the features must come to CPU -- not the other way round
            prior = self.encoder(X_raw[idx.to(X_raw.device)].float().cpu())
        self.prior_z = nn.Parameter(prior.clone())
        self.prior_logp = nn.Parameter(torch.zeros(m.M))

    # ------------------------------------------------------------ plumbing
    def item_embeddings(self, X: torch.Tensor) -> torch.Tensor:
        return self.encoder(X)

    def new_state(self, B: int, device) -> H1State:
        return init_state(B, self.prior_z, self.prior_logp, self.R, device)

    def n_substeps(self, dt_max: float) -> int:
        return n_substeps(dt_max, self.h_max)

    # ------------------------------------------------ step 1: decay / prune
    def decay_prune(self, state: H1State, dt: torch.Tensor, active: torch.Tensor) -> H1State:
        """Decay is continuous (contributions() reads the clock); pruning is the
        discrete part and happens at the event time tau_t = time + dt."""
        if not self.use_satiation:
            return state
        return H1State(z=state.z, logp=state.logp,
                       atoms=prune(state.atoms, state.time + dt, active, self.atom_eps),
                       ema=state.ema, last_item=state.last_item, has_last=state.has_last,
                       time=state.time)

    # ------------------------------------------------------ step 2: evolve
    def evolve(self, state: H1State, E_all: torch.Tensor, dt: torch.Tensor,
               active: torch.Tensor, J: int, want_dis: bool = False):
        """Decay (via the field clock) + transport + replicator.  Must not touch
        observation fields: the returned state carries the SAME atom / ema /
        last_item tensor objects, which event_step verifies."""
        last_emb = E_all[state.last_item] * state.has_last.unsqueeze(-1).to(E_all.dtype)
        ctx = self.context.context_for_interval(last_emb, state.ema, dt)
        u, w = self.potential.emit(ctx)
        field = AngularField(self.potential, u, w, E_all[state.atoms.item], state.atoms,
                             self.gamma, self.h, self.sigma, self.use_satiation)
        p = state.logp.exp()
        t0, t1 = state.time, state.time + dt
        F0 = field.free_energy(state.z, p, t0) if want_dis else None

        z_new = rk4_interval(field, state.z, p, t0, dt, self.eta, J)
        logp_new = replicator_step(state.logp, field.energy_terms(z_new, p, t1),
                                   self.alpha, self.eta, dt)

        a = active
        new = H1State(
            z=torch.where(a.view(-1, 1, 1), z_new, state.z),
            logp=torch.where(a.view(-1, 1), logp_new, state.logp),
            atoms=state.atoms, ema=state.ema, last_item=state.last_item,
            has_last=state.has_last, time=torch.where(a, t1, t0),
        )
        aux = {"field": field}
        if want_dis:
            F1 = field.free_energy(z_new, logp_new.exp(), t1)
            aux["dis_sum"] = ((F1 - F0).clamp_min(0.0) * a.to(F1.dtype)).sum()
        return new, aux

    # ------------------------------------------------------ step 3: heads
    def score(self, state: H1State, E_query: torch.Tensor, E_all: torch.Tensor) -> torch.Tensor:
        return ranking.score(state, E_query, E_all, h=self.h, sigma=self.sigma,
                             use_satiation=self.use_satiation)

    # ------------------------------------------------ step 4: assimilate
    def assimilate(self, state: H1State, item: torch.Tensor, E_all: torch.Tensor,
                   X_all: torch.Tensor, active: torch.Tensor):
        if self.use_satiation:
            a, lam = self.satiation(X_all[item])
            atoms, evicted = deposit(state.atoms, item, a, lam, state.time, active, self.atom_eps)
        else:
            atoms, evicted = state.atoms, torch.zeros_like(active)
        return H1State(
            z=state.z, logp=state.logp, atoms=atoms,
            ema=self.context.assimilate(state.ema, E_all[item], active),
            last_item=torch.where(active, item, state.last_item),
            has_last=state.has_last | active, time=state.time,
        ), evicted

    # ------------------------------------------------------ regularizers
    def ess_penalty(self, state: H1State) -> torch.Tensor:
        """Per-user anti-collapse penalty [B]: (ESS_min - 1/sum p^2)_+^2."""
        return (self.ess_floor - state.ess).clamp_min(0.0).pow(2)

    def reg_ess(self, state: H1State, active: torch.Tensor | None = None) -> torch.Tensor:
        """Anti-collapse floor, inactive while the swarm is healthy.

        CORRECTION C1: the v1.0 form (log M - H[p])_+ never binds (H <= log M
        always), so it was an unconditional push toward uniform masses --
        flattening exactly the structure H1 tests.  At M = 1 this is 0.
        """
        pen = self.ess_penalty(state)
        if active is None:
            return pen.mean()
        w = active.to(pen.dtype)
        return (pen * w).sum() / w.sum().clamp_min(1.0)

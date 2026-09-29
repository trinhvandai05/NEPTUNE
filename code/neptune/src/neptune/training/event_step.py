"""THE single source of truth for event ordering (spec sec. 5.5).

Training, evaluation and the benchmarks all advance state through this one
function.  No other module is allowed to call evolve / assimilate directly, so
the ordering cannot drift between code paths:

    1. decay/prune-- decay is continuous (field clock); atoms below eps at
                     tau_t are pruned here, before anything reads Psi
    2. evolve     -- transport + replicator over the gap
    3. heads      -- read the PRE-observation state
    4. assimilate -- deposit the atom for i_t, update context

Swapping 3 and 4 leaks the label while every metric still looks plausible.
The guard below checks that evolve() and the head leave every observation
tensor untouched -- same object, same in-place version counter.  That check
costs no device synchronization, so it is always on.
"""

from __future__ import annotations

import torch


class OrderingError(RuntimeError):
    """Observation fields changed before the heads were evaluated (label leak)."""


def _fingerprint(state):
    return tuple((id(t), getattr(t, "_version", 0)) for t in state.observation_tensors())


def event_step(model, state, E_all: torch.Tensor, X_all: torch.Tensor, item: torch.Tensor,
               dt: torch.Tensor, active: torch.Tensor, dt_max: float, head_fn=None,
               want_dis: bool = False):
    """Advance one event for the whole batch.

    head_fn(pre_observation_state) -> anything; its return value is passed back.
    Returns (next_state, head_output, aux).
    """
    state = model.decay_prune(state, dt, active)          # step 1 (time-driven only)
    before = _fingerprint(state)
    evolved, aux = model.evolve(state, E_all, dt, active, model.n_substeps(dt_max), want_dis=want_dis)
    if _fingerprint(evolved) != before:
        raise OrderingError(
            "evolve() modified observation fields (atoms / context) before the heads ran. "
            "The current event would leak into its own prediction (spec 5.5).")
    out = head_fn(evolved) if head_fn is not None else None
    if _fingerprint(evolved) != before:
        raise OrderingError("the head modified observation fields in place -- label leak (spec 5.5).")
    nxt, evicted = model.assimilate(evolved, item, E_all, X_all, active)
    aux["evicted"] = evicted
    return nxt, out, aux

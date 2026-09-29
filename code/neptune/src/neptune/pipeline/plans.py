"""Which runs each stage launches -- derived from the preregistered grids only.

Keeping the run lists here (not in the scripts) means the tests exercise the
exact lists the scripts execute, and the run counts can be asserted (v1.2 grid:
6 M, 3 h, 3 sigma; primary arms M in {1,16} x 5 confirmatory seeds, the other
4 arms x 2 seeds):
  shared sigma : F01 18 + F02-seed0 12 + confirmatory 18 = 48 runs
  per-arm sigma: F01 18 + F02-seed0 36 + confirmatory 18 = 72 runs
  robustness (optional): log-dt 10 + popularity negatives 10
"""

from __future__ import annotations

from .selection import candidate_grid


def _neptune(M, h, s, seed, neg, **extra):
    return {"model.M": int(M), "model.h": float(h), "model.sigma": float(s), "seed": int(seed),
            "train.negatives": neg, **extra}


def f01_sweeps(prereg) -> list:
    g = prereg.grids
    return [_neptune(M, h, s, seed, g["negatives_primary"])
            for M in g["f01_M"] for s in g["sigma"] for h in g["h"] for seed in g["seeds_tuning"]]


def f02_seed0_sweeps(prereg, policy: dict) -> list:
    """Remaining arms at seed 0.  F01 arms are included too: under the shared
    policy their shared-sigma runs already exist and are reused, not rerun."""
    g = prereg.grids
    cands = candidate_grid(policy, g["h"], g["sigma"])
    return [_neptune(M, h, s, seed, g["negatives_primary"])
            for M in g["M"] for (h, s) in cands for seed in g["seeds_tuning"]]


def confirmatory_sweeps(prereg, selection: dict) -> list:
    """Confirmatory seeds only -- never the tuning seed; more seeds for the primary arms."""
    g = prereg.grids
    return [_neptune(M, selection["arms"][str(int(M))]["h"], selection["arms"][str(int(M))]["sigma"],
                     seed, g["negatives_primary"])
            for M in g["M"] for seed in prereg.seeds_confirmatory(M)]


def robustness_sweeps(prereg, selection: dict) -> list:
    """log1p(clipped days) time (NOT raw Delta_t) at the frozen h*(M); no re-tuning."""
    g = prereg.grids
    return [_neptune(M, selection["arms"][str(int(M))]["h"], selection["arms"][str(int(M))]["sigma"],
                     seed, g["negatives_primary"], **{"data.ordinal_time": False})
            for M in g["robustness_M"] for seed in prereg.seeds_confirmatory(M)]


def secondary_negatives_sweeps(prereg, selection: dict) -> list:
    """Popularity-sampled negatives at the frozen h*(M): reported next to uniform."""
    g = prereg.grids
    other = [n for n in g["negatives"] if n != g["negatives_primary"]]
    return [_neptune(M, selection["arms"][str(int(M))]["h"], selection["arms"][str(int(M))]["sigma"],
                     seed, neg)
            for neg in other for M in g["robustness_M"] for seed in prereg.seeds_confirmatory(M)]


def baseline_tuning_sweeps(prereg, kind: str, *, dropout=None, sigma_shared=None) -> list:
    """Equal budget: 3 trials each.  SASRec tunes dropout; SASRec+Psi tunes sigma
    at SASRec's selected dropout."""
    g, seed = prereg.grids, prereg.grids["seeds_tuning"][0]
    if kind == "sasrec":
        return [{"baseline.kind": "sasrec", "baseline.dropout": float(do), "seed": int(seed),
                 "model.sigma": float(sigma_shared if sigma_shared is not None else g["sigma"][0])}
                for do in prereg.baselines["sasrec_dropout_grid"]]
    if kind == "sasrec_psi":
        return [{"baseline.kind": "sasrec_psi", "baseline.dropout": float(dropout),
                 "model.sigma": float(s), "seed": int(seed)} for s in g["sigma"]]
    raise ValueError(kind)

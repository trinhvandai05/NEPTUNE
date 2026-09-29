"""Measurement error of lambda_hat (amendment A1.1).

Regressing Delta_u on a noisy lambda_hat attenuates the slope by the
reliability ratio  Var(lambda) / (Var(lambda) + tau^2).  tau^2 ~ O(1/n), so the
attenuation is strongest for SHORT histories -- which produces exactly the
"effect only in long histories" shape that D2-C is meant to rule out.  The two
explanations are indistinguishable unless reliability is measured.

tau^2 is estimated by a SESSION-BLOCK bootstrap.  A multinomial bootstrap
treats n events as n independent draws; a user who binges ten films of one
genre in one evening did not provide ten independent observations.  The
multinomial version is kept only to compute the design effect
DEFF = tau^2_block / tau^2_multinomial, which is reported per stratum because
it may itself vary with history length.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .simpson import unbiased_collision


def multinomial_tau2(counts: np.ndarray, n_boot: int, rng: np.random.Generator) -> float:
    counts = np.asarray(counts, dtype=np.int64)
    n = int(counts.sum())
    draws = rng.multinomial(n, counts / n, size=n_boot)
    return float(np.var(unbiased_collision(draws), ddof=1))


def block_bootstrap_tau2(session_counts: np.ndarray, n_boot: int,
                         rng: np.random.Generator) -> float:
    """session_counts: [S, K] cluster counts per session.  Resample SESSIONS.

    Returns NaN for a single-session history: one block carries no information
    about between-block variability.
    """
    sc = np.asarray(session_counts, dtype=np.float64)
    S = sc.shape[0]
    if S < 2:
        return float("nan")
    idx = rng.integers(0, S, size=(n_boot, S))
    W = np.zeros((n_boot, S))
    np.add.at(W, (np.repeat(np.arange(n_boot), S), idx.ravel()), 1.0)
    draws = W @ sc                                  # [n_boot, K]
    ok = draws.sum(axis=1) >= 2
    if ok.sum() < 2:
        return float("nan")
    return float(np.var(unbiased_collision(draws[ok]), ddof=1))


def reliability_by_stratum(lam: np.ndarray, tau2: np.ndarray, strata: np.ndarray,
                           deff: np.ndarray | None = None) -> pd.DataFrame:
    """R_j = max(0, 1 - mean(tau2_j) / Var_obs(lambda_hat_j)).

    Var_obs already CONTAINS measurement error (Var_obs = Var_true + mean tau2),
    so 1 - mean_tau2/Var_obs is the standard reliability ratio, not a biased
    variant.  It is a diagnostic, not an exact latent reliability for every
    data-generating process.
    """
    rows = []
    for j in np.unique(strata):
        m = strata == j
        lj, tj = lam[m], tau2[m]
        ok = np.isfinite(tj)
        var_obs = float(np.var(lj, ddof=1)) if m.sum() > 1 else float("nan")
        mean_tau2 = float(np.mean(tj[ok])) if ok.any() else float("nan")
        R = max(0.0, 1.0 - mean_tau2 / var_obs) if var_obs and np.isfinite(var_obs) and var_obs > 0 and np.isfinite(mean_tau2) else float("nan")
        row = {"stratum": int(j), "n": int(m.sum()), "var_lambda_obs": var_obs,
               "mean_tau2": mean_tau2, "R": R, "n_tau2_missing": int((~ok).sum())}
        if deff is not None:
            d = deff[m]
            d = d[np.isfinite(d)]
            row["deff_median"] = float(np.median(d)) if len(d) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)

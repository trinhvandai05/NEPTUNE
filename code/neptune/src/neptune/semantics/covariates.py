"""Per-user covariates for the H1 analysis.  Built once, frozen, before training.

Columns
  u, n_train, log_n, n_sessions         history size (train split only)
  lambda_hat                             A1 primary regressor
  tau2_block, tau2_multinomial, deff     A1.1 measurement error
  lambda_A, lambda_B, n_A, n_B           A1.2 session split-half
  tail_train, tail_test                  A1.3 confound (pre- and post-treatment)
  n_val, n_test                          evaluation counts
  length_stratum (1..S)                  quantile of n_train
  lambda_quintile (1..S)                 quantile of lambda_hat; 1 = most diverse
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..common import quantile_labels
from .popularity import head_mask_from_train, tail_fraction
from .reliability import block_bootstrap_tau2, multinomial_tau2
from .simpson import unbiased_collision
from .split_half import session_split_half

TRAIN, VAL, TEST = 0, 1, 2


def _safe_corr(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def compute_user_covariates(events: pd.DataFrame, item_cluster: np.ndarray, n_items: int, *,
                            head_fraction: float, n_boot: int, n_strata: int, seed: int):
    rng = np.random.default_rng(seed)
    ev = events.sort_values(["u", "t"], kind="mergesort")
    train = ev[ev["split"] == TRAIN]
    head = head_mask_from_train(train["i"].to_numpy(), n_items, head_fraction)

    u_arr = train["u"].to_numpy()
    i_arr = train["i"].to_numpy()
    c_arr = item_cluster[i_arr]
    s_arr = train["session"].to_numpy()
    bounds = np.flatnonzero(np.r_[True, u_arr[1:] != u_arr[:-1], True])

    rows = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        uniq_c, c_idx = np.unique(c_arr[a:b], return_inverse=True)
        uniq_s, s_idx = np.unique(s_arr[a:b], return_inverse=True)
        sc = np.zeros((len(uniq_s), len(uniq_c)), dtype=np.int64)
        np.add.at(sc, (s_idx, c_idx), 1)
        counts = sc.sum(axis=0)
        lam = unbiased_collision(counts)            # asserts n >= 2
        lam_a, lam_b, na, nb = session_split_half(sc)
        rows.append((
            int(u_arr[a]), int(b - a), int(len(uniq_s)), lam,
            block_bootstrap_tau2(sc, n_boot, rng), multinomial_tau2(counts, n_boot, rng),
            lam_a, lam_b, na, nb, tail_fraction(i_arr[a:b], head),
        ))
    cov = pd.DataFrame(rows, columns=[
        "u", "n_train", "n_sessions", "lambda_hat", "tau2_block", "tau2_multinomial",
        "lambda_A", "lambda_B", "n_A", "n_B", "tail_train"])
    cov["log_n"] = np.log(cov["n_train"].astype(np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        cov["deff"] = cov["tau2_block"] / cov["tau2_multinomial"]

    te = ev[ev["split"] == TEST]
    tstats = te.groupby("u")["i"].agg(n_test="size",
                                      tail_test=lambda x: tail_fraction(x.to_numpy(), head))
    vstats = ev[ev["split"] == VAL].groupby("u")["i"].agg(n_val="size")
    cov = cov.merge(tstats, left_on="u", right_index=True, how="left")
    cov = cov.merge(vstats, left_on="u", right_index=True, how="left")
    cov["n_test"] = cov["n_test"].fillna(0).astype(np.int64)
    cov["n_val"] = cov["n_val"].fillna(0).astype(np.int64)
    cov["length_stratum"] = quantile_labels(cov["n_train"].to_numpy(), n_strata) + 1
    cov["lambda_quintile"] = quantile_labels(cov["lambda_hat"].to_numpy(), n_strata) + 1

    info = {
        "n_users": int(len(cov)),
        "n_head_items": int(head.sum()),
        "corr_lambda_logn": _safe_corr(cov["lambda_hat"], cov["log_n"]),
        "n_single_session_users": int((cov["n_sessions"] < 2).sum()),
        "n_split_half_usable": int(cov[["lambda_A", "lambda_B"]].notna().all(axis=1).sum()),
        "deff_median": float(np.nanmedian(cov["deff"])) if cov["deff"].notna().any() else float("nan"),
    }
    return cov, head, info

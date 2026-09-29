"""Global temporal split and the evaluation cohort.

T_cut and T_val are quantiles of ALL positive events, computed before any user
is sampled, so the split boundary does not depend on which users are drawn.

Evaluation cohort (amendment A4).  Under a global temporal split a user whose
whole history predates T_val contributes nothing to test evaluation.  In
MovieLens that is a large share of users.  Left unfiltered, the effective N of
every D2 bin collapses silently.  So users are sampled only from those with
  n_train >= min_train_events  AND  n_test >= min_test_events (= 3).
This is a selection on late activity and must be reported as a caveat; the
attrition counts are written to the manifest for exactly that reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .sessions import assign_sessions

TRAIN, VAL, TEST = 0, 1, 2


@dataclass
class SplitResult:
    events: pd.DataFrame      # u:int32, i:int32, t:float64, split:int8, session:int32
    user_ids: np.ndarray      # original MovieLens userId of each u
    t_cut: float
    t_val: float
    attrition: dict


def temporal_split(ev: pd.DataFrame, *, n_users: int, train_frac: float, val_frac: float,
                   min_train_events: int, min_test_events: int,
                   session_gap_seconds: float, seed: int) -> SplitResult:
    t_all = ev["t"].to_numpy()
    t_cut = float(np.quantile(t_all, train_frac))
    t_val = float(np.quantile(t_all, train_frac + val_frac))
    if not t_cut < t_val:
        raise ValueError(f"degenerate split: T_cut={t_cut} >= T_val={t_val}")

    flags = pd.DataFrame({"userId": ev["userId"].to_numpy(),
                          "tr": t_all < t_cut, "te": t_all >= t_val})
    agg = flags.groupby("userId").agg(n_train=("tr", "sum"), n_test=("te", "sum"))
    base = agg["n_train"] >= min_train_events
    att = {
        "users_total": int(len(agg)),
        "users_train_ge_min": int(base.sum()),
        "users_test_ge_1": int((base & (agg["n_test"] >= 1)).sum()),
        "users_test_ge_2": int((base & (agg["n_test"] >= 2)).sum()),
        "users_test_ge_3": int((base & (agg["n_test"] >= 3)).sum()),
        "min_train_events": int(min_train_events),
        "min_test_events": int(min_test_events),
    }
    evaluable = agg.index[base & (agg["n_test"] >= min_test_events)].to_numpy()
    att["users_evaluable"] = int(len(evaluable))
    att["users_excluded_by_test_rule"] = att["users_train_ge_min"] - att["users_evaluable"]

    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(evaluable, size=min(n_users, len(evaluable)), replace=False))
    att["users_sampled"] = int(len(keep))
    # A shortfall is reported in attrition["users_sampled"]; prepare_dataset() refuses it.

    sub = ev[ev["userId"].isin(keep)].sort_values(["userId", "t"], kind="mergesort")
    u = np.searchsorted(keep, sub["userId"].to_numpy()).astype(np.int32)
    t = sub["t"].to_numpy()
    split = np.where(t < t_cut, TRAIN, np.where(t < t_val, VAL, TEST)).astype(np.int8)
    events = pd.DataFrame({
        "u": u, "i": sub["item"].to_numpy().astype(np.int32), "t": t,
        "split": split, "session": assign_sessions(u, t, session_gap_seconds),
    })
    return SplitResult(events=events, user_ids=keep, t_cut=t_cut, t_val=t_val, attrition=att)

import numpy as np
import pandas as pd

from neptune.data.sessions import assign_sessions
from neptune.data.split import temporal_split


def _events(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(300):
        start = rng.uniform(0, 500)
        # 70% stay active into the test window; the rest stop early (exercises attrition)
        end = rng.uniform(900, 1000) if rng.random() < 0.7 else start + rng.uniform(50, 300)
        for t in np.sort(rng.uniform(start, end, rng.integers(15, 60))):
            rows.append((u + 1000, int(rng.integers(0, 50)), float(t)))
    return pd.DataFrame(rows, columns=["userId", "item", "t"])


def test_split_boundaries_and_cohort():
    ev = _events()
    sp = temporal_split(ev, n_users=100, train_frac=0.8, val_frac=0.05, min_train_events=5,
                        min_test_events=3, session_gap_seconds=10.0, seed=0)
    e = sp.events
    assert np.isclose(sp.t_cut, np.quantile(ev["t"], 0.8)) and sp.t_cut < sp.t_val
    assert (e.loc[e["split"] == 0, "t"] < sp.t_cut).all()
    assert (e.loc[e["split"] == 2, "t"] >= sp.t_val).all()
    per = e.groupby("u").agg(n_train=("split", lambda s: (s == 0).sum()),
                             n_test=("split", lambda s: (s == 2).sum()))
    assert (per["n_test"] >= 3).all() and (per["n_train"] >= 5).all()
    a = sp.attrition
    assert a["users_test_ge_1"] >= a["users_test_ge_2"] >= a["users_test_ge_3"] >= a["users_evaluable"]
    assert a["users_sampled"] == e["u"].nunique() == 100
    assert a["users_excluded_by_test_rule"] == a["users_train_ge_min"] - a["users_evaluable"] > 0
    assert (e.groupby("u")["t"].apply(lambda s: s.is_monotonic_increasing)).all()


def test_split_boundary_independent_of_sample():
    ev = _events()
    kw = dict(train_frac=0.8, val_frac=0.05, min_train_events=5, min_test_events=3, session_gap_seconds=10.0)
    a = temporal_split(ev, n_users=50, seed=0, **kw)
    b = temporal_split(ev, n_users=120, seed=1, **kw)
    assert a.t_cut == b.t_cut and a.t_val == b.t_val


def test_sessions():
    u = np.array([0, 0, 0, 0, 1, 1, 1])
    t = np.array([0.0, 10.0, 5000.0, 5010.0, 0.0, 4000.0, 4001.0])
    assert assign_sessions(u, t, 1800).tolist() == [0, 0, 1, 1, 0, 1, 1]

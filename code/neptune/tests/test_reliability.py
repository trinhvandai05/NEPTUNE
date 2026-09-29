import numpy as np

from neptune.semantics.reliability import (block_bootstrap_tau2, multinomial_tau2,
                                           reliability_by_stratum)
from neptune.semantics.split_half import session_split_half


def test_block_bootstrap_exceeds_multinomial_for_coherent_sessions():
    """Binge sessions (one cluster each) carry far less information than n
    independent draws; the block bootstrap must see that (DEFF >> 1)."""
    rng = np.random.default_rng(0)
    sc = np.zeros((12, 4), dtype=np.int64)
    for s in range(12):
        sc[s, s % 4] = 6
    tb = block_bootstrap_tau2(sc, 400, rng)
    tm = multinomial_tau2(sc.sum(0), 400, rng)
    assert tb / tm > 3.0


def test_single_session_gives_nan():
    assert np.isnan(block_bootstrap_tau2(np.array([[3, 2, 1]]), 50, np.random.default_rng(0)))


def test_reliability_formula():
    rng = np.random.default_rng(0)
    lam_true = rng.uniform(0.1, 0.9, 4000)
    tau2 = np.where(np.arange(4000) < 2000, 0.02, 0.001)
    lam_obs = lam_true + rng.normal(0, np.sqrt(tau2))
    strata = (np.arange(4000) >= 2000).astype(int)
    rel = reliability_by_stratum(lam_obs, tau2, strata).set_index("stratum")
    var_true = 0.8 ** 2 / 12
    assert abs(rel.loc[0, "R"] - var_true / (var_true + 0.02)) < 0.04
    assert rel.loc[1, "R"] > rel.loc[0, "R"]


def test_split_half_alternates_sessions_not_events():
    sc = np.array([[4, 0], [0, 4], [4, 0], [0, 4]])      # sessions alternate cluster
    la, lb, na, nb = session_split_half(sc)
    assert la == 1.0 and lb == 1.0 and na == nb == 8
    # alternating EVENTS would have given two identical, mixed halves:
    events = np.repeat([0, 1, 0, 1], 4)
    ea = np.bincount(events[0::2], minlength=2)
    eb = np.bincount(events[1::2], minlength=2)
    assert (ea == eb).all()

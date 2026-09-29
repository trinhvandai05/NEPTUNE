import numpy as np
import pandas as pd

from neptune.semantics.covariates import compute_user_covariates


def _events():
    rng = np.random.default_rng(0)
    rows = []
    for u in range(40):
        for k in range(12):
            rows.append((u, int(rng.integers(0, 10)), float(k), 0, k // 3))
        for k in range(4):
            rows.append((u, int(rng.integers(0, 30)), 100.0 + k, 2, 10))
    return pd.DataFrame(rows, columns=["u", "i", "t", "split", "session"])


def test_future_popularity_cannot_change_head():
    ev = _events()
    clusters = np.arange(30) % 5
    kw = dict(head_fraction=0.2, n_boot=20, n_strata=3, seed=0)
    cov1, head1, _ = compute_user_covariates(ev, clusters, 30, **kw)
    boom = pd.DataFrame([(u, 29, 200.0 + k, 2, 11) for u in range(40) for k in range(3)],
                        columns=ev.columns)                       # item 29: huge TEST popularity
    cov2, head2, _ = compute_user_covariates(pd.concat([ev, boom]), clusters, 30, **kw)
    assert (head1 == head2).all() and not head2[29]
    assert np.allclose(cov1["tail_train"], cov2["tail_train"])     # pre-treatment unchanged
    assert (cov2["tail_test"] >= cov1["tail_test"] - 1e-12).all()  # post-treatment reflects test

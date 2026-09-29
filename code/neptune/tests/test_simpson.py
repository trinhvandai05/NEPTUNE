import numpy as np
import pytest

from neptune.semantics.simpson import unbiased_collision


def test_units():
    assert unbiased_collision([5, 0, 0]) == 1.0                      # one cluster
    assert unbiased_collision([1, 1, 1, 1]) == 0.0                   # all distinct
    assert np.isclose(unbiased_collision([2, 2]), (2 + 2) / 12)      # 4/(4*3)
    batch = unbiased_collision(np.array([[5, 0], [1, 1]]))
    assert batch.tolist() == [1.0, 0.0]


def test_rejects_n_below_two_and_negative():
    with pytest.raises(ValueError, match="n < 2"):
        unbiased_collision([1, 0, 0])
    with pytest.raises(ValueError):
        unbiased_collision([3, -1])


def test_unbiased_across_n_and_variance_shrinks():
    """p = (.5,.3,.2): lambda = .38 at EVERY n (no drift); variance decreases with n.
    This is the property Shannon plug-in entropy lacks."""
    rng = np.random.default_rng(0)
    p = np.array([0.5, 0.3, 0.2])
    means, variances = [], []
    for n in (4, 10, 40, 160):
        est = unbiased_collision(rng.multinomial(n, p, size=20000))
        means.append(est.mean())
        variances.append(est.var())
    assert np.allclose(means, 0.38, atol=0.006), means
    assert all(a > b for a, b in zip(variances, variances[1:])), variances


def test_shannon_plugin_drifts_where_simpson_does_not():
    """Documents WHY A1 replaced entropy: plug-in entropy rises with n at fixed p."""
    rng = np.random.default_rng(1)
    p = np.full(50, 1 / 50)
    def H(c):
        q = c / c.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            return -(np.where(q > 0, q * np.log(q), 0)).sum(1)
    h_small = H(rng.multinomial(10, p, size=4000)).mean()
    h_big = H(rng.multinomial(400, p, size=4000)).mean()
    assert h_big - h_small > 1.0
    l_small = unbiased_collision(rng.multinomial(10, p, size=4000)).mean()
    l_big = unbiased_collision(rng.multinomial(400, p, size=4000)).mean()
    assert abs(l_big - l_small) < 0.005

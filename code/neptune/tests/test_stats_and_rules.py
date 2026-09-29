import numpy as np
import pandas as pd

from neptune.evaluation.h1_analysis import d2c
from neptune.evaluation.stats import ols_hc3, tsls
from neptune.evaluation.stop_rules import evaluate_acceptance

ACC = {"c5_cv_max": 0.75, "c5_r_min": 0.30}


def test_ols_matches_lstsq_and_hc3_is_sane():
    rng = np.random.default_rng(0)
    x = rng.normal(size=3000)
    y = 1.0 - 2.0 * x + rng.normal(size=3000) * (1 + np.abs(x))
    fit = ols_hc3(y, {"x": x})
    b = np.linalg.lstsq(np.column_stack([np.ones(3000), x]), y, rcond=None)[0]
    assert np.isclose(fit["coef"]["x"]["beta"], b[1])
    assert fit["coef"]["x"]["lo"] < -2.0 < fit["coef"]["x"]["hi"]


def test_iv_undoes_attenuation():
    rng = np.random.default_rng(1)
    n = 6000
    lam = rng.uniform(0.1, 0.9, n)
    a, b = lam + rng.normal(0, 0.2, n), lam + rng.normal(0, 0.2, n)
    y = -1.0 * lam + rng.normal(0, 0.1, n)
    ols = ols_hc3(y, {"a": a})["coef"]["a"]["beta"]
    iv = tsls(y, a, b, {}, n_boot=100, seed=0)
    assert ols > -0.75                                  # attenuated toward 0
    assert abs(iv["beta"] + 1.0) < 0.1 and iv["first_stage_F"] > 100


def _world(beta_by_stratum, tau2_by_stratum, seed=0, n=1500, seeds=(1, 2, 3)):
    rng = np.random.default_rng(seed)
    rows = []
    for j, (bj, t2) in enumerate(zip(beta_by_stratum, tau2_by_stratum), 1):
        lam = rng.uniform(0.1, 0.9, n)
        d = {"length_stratum": j, "lambda_hat": lam + rng.normal(0, np.sqrt(t2), n), "tau2_block": t2,
             "deff": 1.5, "n_train": 10 * j}
        for s in seeds:
            d[f"delta_s{s}"] = bj * lam + rng.normal(0, 0.05, n)
        rows.append(pd.DataFrame(d))
    df = pd.concat(rows, ignore_index=True)
    cols = [f"delta_s{s}" for s in seeds]
    df["delta"] = df[cols].mean(axis=1)
    return df, cols


def _c5(world):
    df, cols = world
    an = {"C1": {"lo": 0.01, "sign_consistent": True},
          "D2B": {k: {"ok": True, "hi": -0.1} for k in ("lambda", "lambda+logn", "lambda+logn+tail_test")},
          "D2C": d2c(df, cols, 0.05, ACC["c5_r_min"])}
    return evaluate_acceptance(an, ACC, {"ok": True})


def test_condition5_passes_when_gradient_is_pure_attenuation():
    """H1 true and uniform; short strata look weaker only because lambda_hat is noisier.
    This world MUST pass -- the draft wording of C5 would have failed it."""
    res = _c5(_world([-1.0] * 4, [0.03, 0.015, 0.006, 0.001]))
    assert res["conditions"]["C5_not_length_artifact"], res["C5_detail"]
    assert res["verdict"] == "SUPPORTED"


def test_condition5_fails_when_effect_lives_only_in_long_histories():
    res = _c5(_world([0.0, 0.0, -1.0, -1.5], [0.001] * 4))
    assert not res["conditions"]["C5_not_length_artifact"]
    assert res["verdict"] == "SUGGESTIVE"


def test_verdict_mapping():
    base = {"C1": {"lo": -0.01, "sign_consistent": True},
            "D2B": {k: {"ok": True, "hi": 0.1} for k in ("lambda", "lambda+logn", "lambda+logn+tail_test")},
            "D2C": {"strata": [], "fe_combined": None, "n_reliable": 0, "ratio_mean": np.nan, "ratio_cv": np.nan}}
    assert evaluate_acceptance(base, ACC, {"ok": True})["verdict"] == "STOP"
    assert evaluate_acceptance(base, ACC, {"ok": True, "D17_no_manifold_collapse": False})["verdict"] == "INVALID_PROTOCOL"
    base["C1"]["lo"] = 0.01
    assert evaluate_acceptance(base, ACC, {"ok": True})["verdict"] == "CAPACITY_ONLY"
    base["C1"]["sign_consistent"] = False                        # a CI alone is not enough
    assert evaluate_acceptance(base, ACC, {"ok": True})["verdict"] == "STOP"

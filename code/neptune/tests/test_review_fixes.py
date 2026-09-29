"""Regression tests for the five blocking issues of the v1.1 external review."""
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from conftest import FIX, tiny_model
from neptune.config import ConfigError, build_run_config, load_prereg
from neptune.evaluation.h1_analysis import primary_contrast, run_h1_analysis
from neptune.logging.registry import run_dir
from neptune.training.event_step import event_step


# --- P0 #1: preregistration versions can never share or reuse runs -----------------
def test_run_dirs_are_namespaced_by_prereg(prereg, profile, tmp_path):
    raw = yaml.safe_load((FIX / "prereg_smoke.yaml").read_text())
    raw["version"], raw["frozen"]["train"]["epochs"] = "9.9.9", 99
    (tmp_path / "p2.yaml").write_text(yaml.safe_dump(raw))
    other = load_prereg(tmp_path / "p2.yaml")
    sweep = {"model.M": 4, "model.h": 0.3, "model.sigma": 0.25, "seed": 0}
    a = run_dir("out", build_run_config(prereg, profile, sweep))
    b = run_dir("out", build_run_config(other, profile, sweep))
    assert a.name == b.name and a != b                  # same key, different namespace


def test_reuse_refuses_a_run_whose_stored_config_differs(prereg, profile, data_dir, tmp_path):
    from conftest import keep_gate
    from neptune.pipeline.runner import execute
    keep_gate(tmp_path, prereg)
    sweep = {"model.M": 1, "model.h": 0.3, "model.sigma": 0.25, "seed": 0}
    cfg = build_run_config(prereg, profile, sweep)
    rdir = run_dir(tmp_path, cfg)
    rdir.mkdir(parents=True)
    tampered = cfg.to_dict()
    tampered["train"]["epochs"] = 99
    (rdir / "config.yaml").write_text(yaml.safe_dump(tampered))
    (rdir / "summary.json").write_text('{"prereg_sha256": "%s", "val": {"metric": "x", "value": 0}}' % cfg.prereg_sha256)
    with pytest.raises(ConfigError, match="refusing to reuse"):
        execute(prereg, profile, sweep, data_dir=data_dir, out_root=tmp_path, device="cpu",
                experiment="F01_BANDWIDTH_PROBE", log=lambda *x: None)


# --- P0 #2 and #3: tuning seed excluded; seed variance enters the CI -------------------
def _results(delta_by_seed, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s, d in delta_by_seed.items():
        base = rng.uniform(0.1, 0.3, n)
        for M, shift in ((1, 0.0), (4, d)):
            rows.append(pd.DataFrame({"u": np.arange(n), "M": M, "seed": s,
                                      "ndcg@10": base + shift + rng.normal(0, 0.05, n)}))
    return pd.concat(rows, ignore_index=True)


def test_seed_variance_widens_the_interval_and_sign_must_agree():
    res = _results({1: 0.02, 2: -0.004, 3: 0.02})
    D, c1 = primary_contrast(res, "ndcg@10", 4, 1, [1, 2, 3], 0.05)
    user_only_lo = c1["est"] - 1.96 * c1["se_user"]
    assert user_only_lo > 0                              # users alone would "prove" it ...
    assert c1["lo"] < 0 and not c1["sign_consistent"]    # ... seeds say otherwise


def test_tuning_seed_never_enters_acceptance():
    res = _results({0: 0.5, 1: 0.01, 2: 0.012})          # seed 0: huge winner's-curse effect
    cov = pd.DataFrame({"u": np.arange(4000), "lambda_hat": np.random.default_rng(1).uniform(0, 1, 4000),
                        "log_n": 3.0, "n_train": 20, "n_test": 5, "tail_test": 0.5, "tail_train": 0.5,
                        "tau2_block": 0.01, "deff": 1.0, "lambda_A": np.nan, "lambda_B": np.nan,
                        "length_stratum": np.arange(4000) % 3 + 1, "lambda_quintile": np.arange(4000) % 3 + 1})
    kw = dict(metric="ndcg@10", treat_M=4, ctrl_M=1, alpha=0.05, r_min=0.3, n_boot_iv=10)
    an = run_h1_analysis(res, cov, seeds_by_arm={1: [1, 2], 4: [1, 2]}, tuning_seeds=[0], **kw)
    assert abs(an["C1"]["est"] - 0.011) < 0.005 and an["D0"]["delta"] > 0.4
    with pytest.raises(ValueError, match="tuning seed"):
        run_h1_analysis(res, cov, seeds_by_arm={1: [0, 1], 4: [0, 1]}, tuning_seeds=[0], **kw)


def test_prereg_rejects_overlapping_tuning_and_confirmatory_seeds(tmp_path):
    raw = yaml.safe_load((FIX / "prereg_smoke.yaml").read_text())
    raw["grids"]["seeds_confirmatory_primary"] = [0, 1]
    (tmp_path / "bad.yaml").write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="tuning seed"):
        load_prereg(tmp_path / "bad.yaml")


# --- P0 #4: atoms that decayed below eps are pruned BEFORE the heads read Psi ---------
def test_prune_happens_before_the_heads(cfg):
    model, X = tiny_model(cfg)
    E = model.item_embeddings(X).detach()
    st = model.new_state(1, "cpu")
    act = torch.ones(1, dtype=torch.bool)
    st, _, _ = event_step(model, st, E, X, torch.tensor([3]), torch.ones(1), act, 1.0)
    assert st.atoms.mask.any()
    seen = {}
    event_step(model, st, E, X, torch.tensor([4]), torch.full((1,), 200.0), act, 200.0,
               head_fn=lambda s: seen.setdefault("alive", bool(s.atoms.mask.any())))
    assert seen["alive"] is False                         # decayed atom gone before scoring


# --- P0 #5: ESS guard covers users whose sequence ends inside the window -------------
def test_ess_penalty_is_accumulated_over_every_active_event(cfg, monkeypatch):
    from neptune.heads.negatives import NegativeSampler
    from neptune.training.trainer import neptune_train_batch
    model, X = tiny_model(cfg, n_items=40)
    monkeypatch.setattr(model, "ess_penalty", lambda st: torch.tensor([1.0, 2.0]))
    items = torch.randint(0, 40, (2, 10))
    mask = torch.zeros(2, 10, dtype=torch.bool)
    mask[0, :3], mask[1, :] = True, True                 # row 0 ends at event 3 of a 16-wide window
    cpu = {"items": items, "dt": torch.ones(2, 10), "mask": mask,
           "split": torch.zeros(2, 10, dtype=torch.int8), "users": torch.tensor([0, 1])}
    opt = torch.optim.AdamW(model.parameters())
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    samp = NegativeSampler(np.ones(40, bool), np.ones(40), "uniform", "cpu", 0)
    st = neptune_train_batch(model, opt, sched, cpu, X, samp, cfg, "cpu")
    assert np.isclose(st.ess_pen, (1.0 * 3 + 2.0 * 10) / 13)   # v1.1 (last column only) gave 2.0

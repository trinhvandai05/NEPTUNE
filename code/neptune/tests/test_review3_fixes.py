"""Regression tests for the third external review (v1.2.1 -> v1.2.2)."""
import shutil

import numpy as np
import pytest
import torch
import yaml

from conftest import FIX, keep_gate, write_fake_pilot
from neptune.common import read_json, write_json_atomic
from neptune.config import ConfigError, Profile, build_run_config, load_prereg, load_profile
from neptune.pipeline.gates import GateError, check_pilot_config, require_open1
from neptune.pipeline.runner import check_implementation, execute
from neptune.provenance import implementation_sha256

QUIET = dict(device="cpu", log=lambda *x: None)


def _pilot_sweep(prereg):
    from neptune.pipeline.gates import pilot_sweeps
    return pilot_sweeps(prereg)[0]


# --- P0 #1: only the canonical pilot can decide OPEN-1 ------------------------------
def test_non_canonical_pilot_is_rejected_before_it_runs(prereg, data_dir, tmp_path):
    tiny = Profile("tiny_gate", False, ["data.n_users", "train.warmup_steps"],
                   {"data.n_users": 10, "train.warmup_steps": 1})
    with pytest.raises(GateError, match="non-canonical"):
        check_pilot_config(prereg, build_run_config(prereg, tiny, _pilot_sweep(prereg)))
    with pytest.raises(GateError, match="non-canonical"):
        execute(prereg, tiny, _pilot_sweep(prereg), data_dir=data_dir, out_root=tmp_path,
                experiment="P00_PILOT", **QUIET)
    assert not list(tmp_path.rglob("summary.json"))
    check_pilot_config(prereg, build_run_config(prereg, load_profile(FIX / "smoke_pilot.yaml"),
                                                _pilot_sweep(prereg)))          # canonical: accepted


# --- P0 #2: the pilot is protocol-critical, so its artifact must match exactly --------
def test_pilot_refuses_a_stale_artifact(prereg, data_dir, tmp_path):
    stale = tmp_path / "stale"
    shutil.copytree(data_dir, stale)
    man = read_json(stale / "manifest.json")
    write_json_atomic(stale / "manifest.json", {**man, "prereg_sha256": "OLD_PREREG"})
    with pytest.raises(ConfigError, match="does not belong"):
        execute(prereg, load_profile(FIX / "smoke_pilot.yaml"), _pilot_sweep(prereg),
                data_dir=stale, out_root=tmp_path / "out", experiment="P00_PILOT", **QUIET)


def test_gate_rechecks_every_piece_of_evidence(prereg, tmp_path):
    keep_gate(tmp_path, prereg)
    man = tmp_path / "fake_pilot_data" / "manifest.json"
    write_json_atomic(man, {**read_json(man), "note": "rebuilt later"})
    with pytest.raises(GateError, match="data_manifest"):
        require_open1(prereg, tmp_path)


# --- P0 #3: code is locked by an implementation fingerprint --------------------------
def _locked(tmp_path, sha):
    raw = yaml.safe_load((FIX / "prereg_smoke.yaml").read_text())
    raw["implementation"] = {"sha256": sha}
    (tmp_path / "locked.yaml").write_text(yaml.safe_dump(raw))
    return load_prereg(tmp_path / "locked.yaml")


def test_protocol_runs_refuse_other_code(tmp_path):
    wrong = _locked(tmp_path, "0" * 64)
    with pytest.raises(ConfigError, match="code changed after freezing"):
        check_implementation(wrong, strict=True)
    assert check_implementation(wrong, strict=False) == implementation_sha256()   # exploration: allowed
    right = _locked(tmp_path, implementation_sha256())
    assert check_implementation(right, strict=True) == implementation_sha256()


def test_reuse_refuses_a_run_made_by_other_code(prereg, data_dir, tmp_path):
    prof, sw = load_profile(FIX / "smoke_pilot.yaml"), _pilot_sweep(prereg)
    kw = dict(data_dir=data_dir, out_root=tmp_path, experiment="P00_PILOT", **QUIET)
    execute(prereg, prof, sw, **kw)                                 # real (tiny) pilot run
    prov_p = next(tmp_path.rglob("provenance.json"))
    assert read_json(prov_p)["protocol_critical"] is True
    execute(prereg, prof, sw, **kw)                                 # identical code: reused
    write_json_atomic(prov_p, {**read_json(prov_p), "implementation_sha256": "f" * 64})
    with pytest.raises(ConfigError, match="different code"):
        execute(prereg, prof, sw, **kw)


def test_analysis_never_mixes_code_versions():
    from neptune.pipeline.analyze import implementation_consistent
    assert implementation_consistent({"a"}, None) and implementation_consistent({"a"}, "a")
    assert not implementation_consistent({"a", "b"}, None)          # commit A + commit B runs
    assert not implementation_consistent({"a"}, "b") and not implementation_consistent({None}, None)


def test_shipped_preregistration_matches_shipped_code(real_prereg):
    assert real_prereg.implementation_sha256 == implementation_sha256(), (
        "code changed after freezing: re-freeze implementation.sha256 (scripts/fingerprint.py) "
        "under a new preregistration version")


# --- P1 #4: a short cohort is fatal, not a warning ----------------------------------
def test_cohort_shortfall_is_fatal(prereg, profile, tmp_path):
    from neptune.pipeline.prepare import prepare_dataset
    from neptune.synthetic import write_synthetic_ml25m
    raw = write_synthetic_ml25m(tmp_path / "raw", n_users=60)
    cfg = build_run_config(prereg, profile, for_training=False)
    with pytest.raises(ConfigError, match="cohort shortfall"):
        prepare_dataset(raw, tmp_path / "data", cfg)
    assert (tmp_path / "data" / "cohort_shortfall.json").exists()
    assert not (tmp_path / "data" / "manifest.json").exists()      # unusable by construction


# --- P1 #5: D17 says what it measures; spread diagnostics catch what it cannot ---------
def test_d17_mean_cos_is_blind_to_antipodal_collapse_but_diagnostics_are_not():
    from neptune.training.trainer import item_manifold_stats

    class Fake(torch.nn.Module):
        def item_embeddings(self, X):
            v = torch.zeros(len(X), 64)
            v[:, 0] = torch.where(torch.arange(len(X)) % 2 == 0, 1.0, -1.0)
            return v
    st = item_manifold_stats(Fake(), torch.zeros(1000, 3), n_pairs=20000)
    assert abs(st["mean_pairwise_cos"]) < 0.01                      # passes the 0.60 gate ...
    assert st["mean_abs_cos"] > 0.99 and st["effective_rank"] < 1.01  # ... but is 1-dimensional
    assert st["cos_q05"] < -0.99 and st["cos_q95"] > 0.99

"""Regression tests for the fourth external review (v1.2.2 -> v1.2.3)."""
import shutil

import numpy as np
import pandas as pd
import pytest
import yaml

from conftest import FIX, write_fake_pilot
from neptune.common import read_json, write_json_atomic
from neptune.config import ConfigError, build_run_config, load_prereg
from neptune.pipeline.artifacts import verify_data_payload, verify_semantics
from neptune.pipeline.gates import GateError, decide_open1, freeze_open1, gate_path, require_open1
from neptune.pipeline.selection import (anchor_selection, collect_val_table, decide_sigma_policy,
                                        freeze_policy, freeze_selection, select_per_arm,
                                        verify_frozen_selection)


def _locked_wrong(tmp_path):
    raw = yaml.safe_load((FIX / "prereg_smoke.yaml").read_text())
    raw["implementation"] = {"sha256": "0" * 64}
    (tmp_path / "locked.yaml").write_text(yaml.safe_dump(raw))
    return load_prereg(tmp_path / "locked.yaml")


# --- P0 #1: the OPEN-1 verdict is re-derived, never trusted -------------------------
def test_hand_edited_open1_verdict_is_rejected(prereg, tmp_path):
    thr = prereg.acceptance["d17_max_mean_cos"]
    prof = write_fake_pilot(tmp_path, prereg, {0: thr + 0.005})
    freeze_open1(decide_open1(prereg, prof, tmp_path), gate_path(tmp_path, prereg))
    d = read_json(gate_path(tmp_path, prereg))
    write_json_atomic(gate_path(tmp_path, prereg), {**d, "verdict": "KEEP_CURRENT_PREREG"})
    with pytest.raises(GateError, match="inconsistent with its own evidence"):
        require_open1(prereg, tmp_path)
    write_json_atomic(gate_path(tmp_path, prereg), {**d, "verdict": "KEEP_CURRENT_PREREG", "max_D17_final": 0.1})
    with pytest.raises(GateError, match="inconsistent"):
        require_open1(prereg, tmp_path)


# --- P0 #2: the code that decides and analyses is frozen code too --------------------
def test_decision_selection_and_analysis_refuse_other_code(tmp_path, data_dir, profile):
    from neptune.pipeline.analyze import analyze_h1
    wrong = _locked_wrong(tmp_path)
    with pytest.raises(ConfigError, match="code changed after freezing"):
        analyze_h1(wrong, profile, data_dir=data_dir, out_root=tmp_path)      # before touching any file
    with pytest.raises(ConfigError, match="code changed"):
        decide_open1(wrong, write_fake_pilot(tmp_path / "p", wrong), tmp_path / "p")
    with pytest.raises(ConfigError, match="code changed"):
        freeze_policy({"mode": "shared", "shared_sigma": 0.25}, pd.DataFrame(columns=["M", "file", "sha256"]),
                      [1], wrong, tmp_path / "pol.json")
    with pytest.raises(ConfigError, match="code changed"):
        anchor_selection({"arms": {}}, wrong, data_dir)


# --- P0 #3: the H1 moderator (covariates) and the data payload are tamper-evident ------
def test_tampered_covariates_are_detected(prereg, profile, data_dir, tmp_path):
    import dataclasses
    d = tmp_path / "data"
    shutil.copytree(data_dir, d)
    sem_cfg = dataclasses.asdict(build_run_config(prereg, profile, for_training=False).semantics)
    verify_semantics(d, prereg, sem_cfg)                                        # intact: passes
    cov = pd.read_parquet(d / "covariates.parquet")
    cov["lambda_hat"] = cov["lambda_hat"][::-1].to_numpy()                      # scramble the moderator
    cov.to_parquet(d / "covariates.parquet", index=False)
    with pytest.raises(ConfigError, match="modified"):
        verify_semantics(d, prereg, sem_cfg)
    shutil.copy(data_dir / "covariates.parquet", d / "covariates.parquet")
    sem = read_json(d / "semantics_manifest.json")
    write_json_atomic(d / "semantics_manifest.json", {**sem, "implementation_sha256": "x"})
    with pytest.raises(ConfigError, match="implementation_sha256"):
        verify_semantics(d, prereg, sem_cfg)


def test_tampered_data_payload_is_detected(prereg, data_dir, tmp_path):
    d = tmp_path / "data"
    shutil.copytree(data_dir, d)
    verify_data_payload(d, prereg, strict=True)
    ev = pd.read_parquet(d / "events.parquet")
    ev.loc[0, "i"] = (ev.loc[0, "i"] + 1) % 50
    ev.to_parquet(d / "events.parquet", index=False)
    with pytest.raises(ConfigError, match="modified"):
        verify_data_payload(d, prereg, strict=False)


# --- selection: stamped, anchored, re-derived ------------------------------------------
def _fake_tuning_runs(out, prereg, profile):
    from neptune.logging.registry import namespace
    g = prereg.grids
    base = out / "runs" / namespace(profile.name, prereg.sha256)
    for M in g["M"]:
        for h in g["h"]:
            for s in g["sigma"]:
                rd = base / f"M{M}_h{h}_s{s}"
                rd.mkdir(parents=True)
                cfg = {"prereg_sha256": prereg.sha256, "model": {"M": M, "h": h, "sigma": s}, "seed": 0,
                       "train": {"negatives": "uniform"}, "data": {"ordinal_time": True},
                       "baseline": {"kind": "none"}}
                (rd / "config.yaml").write_text(yaml.safe_dump(cfg))
                v = 0.1 + 0.01 * M + (0.02 if h == 0.5 else 0) + (0.004 if s == 0.25 else 0)
                pd.DataFrame({"u": [0, 1], "ndcg@10": [v, v]}).to_parquet(rd / "peruser_val.parquet")


def _freeze_all(out, prereg, profile, data):
    from neptune.logging.registry import selection_dir
    g, tun = prereg.grids, prereg.tuning
    sdir = selection_dir(out, profile.name, prereg.sha256)
    val = collect_val_table(out, profile.name, prereg.sha256, metric="ndcg@10")
    pol = freeze_policy(decide_sigma_policy(val[val["M"].isin(g["f01_M"])], g["f01_M"], g["sigma"], g["h"],
                                            tun["sigma_cost_threshold"]), val, g["f01_M"], prereg,
                        sdir / "sigma_policy.json")
    sel = select_per_arm(val, pol, g["M"], g["h"], g["sigma"], "ndcg@10")
    freeze_selection(anchor_selection(sel, prereg, data), sdir / "selected_h.json")
    return sdir


def test_frozen_selection_is_verified_and_rederived(prereg, profile, data_dir, tmp_path):
    out, data = tmp_path / "out", tmp_path / "data"
    shutil.copytree(data_dir, data)
    _fake_tuning_runs(out, prereg, profile)
    sdir = _freeze_all(out, prereg, profile, data)
    assert verify_frozen_selection(prereg, out, profile.name, data)["arms"]["4"]["h"] == 0.5
    # 1) hand-edit h*(M): sources intact, but re-derivation disagrees
    sel = read_json(sdir / "selected_h.json")
    bad = {**sel, "arms": {**sel["arms"], "4": {**sel["arms"]["4"], "h": 0.3}}}
    write_json_atomic(sdir / "selected_h.json", bad)
    with pytest.raises(RuntimeError, match="re-derived"):
        verify_frozen_selection(prereg, out, profile.name, data)
    write_json_atomic(sdir / "selected_h.json", sel)
    # 2) a cited validation file changes
    f = sel["source_files"][0]["file"]
    pd.DataFrame({"u": [0, 1], "ndcg@10": [0.9, 0.9]}).to_parquet(f)
    with pytest.raises(RuntimeError, match="changed or vanished"):
        verify_frozen_selection(prereg, out, profile.name, data)


def test_covariates_cannot_change_after_selection_is_frozen(prereg, profile, data_dir, tmp_path):
    out, data = tmp_path / "out", tmp_path / "data"
    shutil.copytree(data_dir, data)
    _fake_tuning_runs(out, prereg, profile)
    _freeze_all(out, prereg, profile, data)
    sem = read_json(data / "semantics_manifest.json")
    write_json_atomic(data / "semantics_manifest.json", {**sem, "rebuilt": True})   # covariates rebuilt later
    with pytest.raises(RuntimeError, match="covariates changed after"):
        verify_frozen_selection(prereg, out, profile.name, data)

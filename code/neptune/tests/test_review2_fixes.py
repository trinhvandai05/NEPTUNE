"""Regression tests for the second external review (v1.2 -> v1.2.1)."""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from conftest import keep_gate, write_fake_pilot
from neptune.config import ConfigError, build_run_config
from neptune.pipeline.artifacts import check_artifacts_match, load_artifacts
from neptune.pipeline.gates import (ADOPT, KEEP, GateError, decide_open1, freeze_open1, gate_path,
                                    require_open1)


# --- P0 #1: artifacts from another preregistration / code schema are refused ----------
def test_stale_artifact_refused_for_claim_runs(data_dir, prereg, profile):
    art = load_artifacts(data_dir)
    cfg = build_run_config(prereg, profile, {"model.M": 1, "model.h": 0.3, "model.sigma": 0.25})
    assert check_artifacts_match(art, cfg) == {}
    for key, bad in (("prereg_sha256", "OLD_PREREG"), ("schema", "neptune-data/1.1")):
        stale = dataclasses.replace(art, manifest={**art.manifest, key: bad})
        with pytest.raises(ConfigError, match="rebuild"):
            check_artifacts_match(stale, cfg)                        # same data_cfg, still refused
        pilot = dataclasses.replace(cfg, claim_eligible=False)
        assert key in check_artifacts_match(stale, pilot)            # non-claim: reported, not fatal


# --- P0 #2: OPEN-1 is a hard gate ---------------------------------------------------
def test_claim_runs_refuse_to_start_without_a_keep_decision(prereg, profile, data_dir, tmp_path):
    from neptune.pipeline.runner import execute
    sweep = {"model.M": 1, "model.h": 0.3, "model.sigma": 0.25, "seed": 0}
    kw = dict(data_dir=data_dir, out_root=tmp_path, device="cpu", experiment="F01_BANDWIDTH_PROBE",
              log=lambda *x: None)
    with pytest.raises(GateError, match="not been decided"):
        execute(prereg, profile, sweep, **kw)
    thr = prereg.acceptance["d17_max_mean_cos"]
    prof = write_fake_pilot(tmp_path, prereg, {0: thr + 0.005})      # one pilot run above the gate
    dec = freeze_open1(decide_open1(prereg, prof, tmp_path), gate_path(tmp_path, prereg))
    assert dec["verdict"] == ADOPT
    with pytest.raises(GateError, match="ADOPT_V1_3"):
        execute(prereg, profile, sweep, **kw)
    assert not list((tmp_path / "runs" / profile.name).rglob("summary.json"))   # nothing trained


def test_keep_decision_is_write_once_and_tamper_evident(prereg, tmp_path):
    dec = keep_gate(tmp_path, prereg)
    assert dec["verdict"] == KEEP and require_open1(prereg, tmp_path)["verdict"] == KEEP
    prof = write_fake_pilot(tmp_path, prereg, {1: 0.99})             # pilot evidence rewritten later
    with pytest.raises(GateError, match="changed after"):
        require_open1(prereg, tmp_path)
    with pytest.raises(GateError, match="refusing to overwrite"):
        freeze_open1(decide_open1(prereg, prof, tmp_path), gate_path(tmp_path, prereg))


# --- P1 #3: the decision reads only this preregistration's pilot ----------------------
def test_open1_reads_only_its_own_namespace(prereg, tmp_path):
    import yaml
    from neptune.config import load_prereg
    from conftest import FIX
    raw = yaml.safe_load((FIX / "prereg_smoke.yaml").read_text())
    raw["version"] = "0.0.0-other"
    (tmp_path / "other.yaml").write_text(yaml.safe_dump(raw))
    other = load_prereg(tmp_path / "other.yaml")
    write_fake_pilot(tmp_path, other, {k: 0.99 for k in range(4)})   # collapsed, other prereg
    with pytest.raises(GateError, match="missing"):                  # ... is NOT picked up
        decide_open1(prereg, write_fake_pilot(tmp_path / "x", prereg), tmp_path)
    prof = write_fake_pilot(tmp_path, prereg)
    assert decide_open1(prereg, prof, tmp_path)["verdict"] == KEEP


def test_pilot_must_train_the_claim_epoch_count(prereg, tmp_path):
    from neptune.config import Profile
    short = Profile("short_pilot", False, ["train.epochs"], {"train.epochs": 0})
    with pytest.raises(GateError, match="epoch"):
        write_fake_pilot(tmp_path, prereg)                            # files exist under smoke_pilot ...
        decide_open1(prereg, short, tmp_path)                         # ... but this pilot trains 0 epochs


def test_single_d17_threshold(real_prereg):
    assert real_prereg.acceptance["d17_max_mean_cos"] == 0.60 and real_prereg.gates["open1_required"]


# --- P1 #4: D1 compares every arm with the control on the SAME seeds -------------------
def test_d1_uses_matched_control_seeds():
    from neptune.evaluation.h1_analysis import d1_table
    rng = np.random.default_rng(0)
    rows = []
    for s in (1, 2, 3, 4, 5):
        base = rng.uniform(0.1, 0.3, 500)
        rows.append(pd.DataFrame({"u": np.arange(500), "M": 1, "seed": s,
                                  "ndcg@10": base + (0.1 if s >= 3 else 0.0)}))   # seeds 3-5 of M=1 high
        if s <= 2:
            rows.append(pd.DataFrame({"u": np.arange(500), "M": 4, "seed": s, "ndcg@10": base}))
    res = pd.concat(rows)
    d1 = {r["M"]: r for r in d1_table(res, "ndcg@10", {1: [1, 2, 3, 4, 5], 4: [1, 2]}, 1, 0.05, None, 99)}
    assert abs(d1[4]["delta_vs_ctrl"]) < 1e-12                        # v1.2 gave -0.06


# --- P1 #5: training keeps the full train history ------------------------------------
def test_full_train_history_is_used(real_prereg):
    from neptune.training.batching import BucketedBatcher, Sequences
    assert real_prereg.frozen["train.max_len"] == 0
    seqs = Sequences(items=[np.arange(3000)], times=[np.arange(3000.0)], split=[np.zeros(3000, np.int8)],
                     users=np.array([0]))
    b = BucketedBatcher(seqs, 4, mode="train", ordinal=True, max_len=0)
    assert int(b.batch(0)["mask"].sum()) == 3000

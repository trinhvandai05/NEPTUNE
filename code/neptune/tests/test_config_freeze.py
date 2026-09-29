import pytest

from neptune.config import ConfigError, Profile, build_run_config, load_profile


def test_claim_profile_loads_frozen_values(real_prereg):
    prof = load_profile("configs/profiles/h1_claim.yaml")
    c = build_run_config(real_prereg, prof, {"model.M": 16, "model.h": 0.35, "model.sigma": 0.30, "seed": 0})
    assert c.claim_eligible and c.train.warmup_steps == 2000 and c.data.min_test_events == 3
    assert c.prereg_sha256 == real_prereg.sha256


def test_frozen_key_cannot_be_swept(real_prereg):
    prof = load_profile("configs/profiles/h1_claim.yaml")
    with pytest.raises(ConfigError, match="frozen"):
        build_run_config(real_prereg, prof, {"train.warmup_steps": 500})
    with pytest.raises(ConfigError, match="frozen"):
        build_run_config(real_prereg, prof, {"data.min_test_events": 1})


def test_claim_profile_cannot_override(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("name: bad\nclaim_eligible: true\nmay_override: [train.epochs]\noverrides: {train: {epochs: 2}}\n")
    with pytest.raises(ConfigError, match="claim-eligible"):
        load_profile(p)


def test_profile_override_must_be_whitelisted(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("name: bad\nclaim_eligible: false\nmay_override: [train.epochs]\noverrides: {train: {lr: 0.1}}\n")
    with pytest.raises(ConfigError, match="may_override"):
        load_profile(p)


def test_pilot_is_never_claim_eligible(real_prereg):
    c = build_run_config(real_prereg, load_profile("configs/profiles/engineering_pilot.yaml"), {"model.M": 1})
    assert not c.claim_eligible and c.train.warmup_steps == 500
    with pytest.raises(ConfigError):
        c.assert_claim_eligible()


def test_claim_run_must_be_on_grid(real_prereg):
    prof = load_profile("configs/profiles/h1_claim.yaml")
    with pytest.raises(ConfigError, match="grid"):
        build_run_config(real_prereg, prof, {"model.M": 3})
    with pytest.raises(ConfigError, match="seed"):
        build_run_config(real_prereg, prof, {"model.M": 1, "model.h": 0.2, "model.sigma": 0.2, "seed": 7})


def test_unknown_key_and_type_errors(real_prereg):
    prof = Profile("p", False, ["model.*"], {"model.bogus": 1})
    with pytest.raises(ConfigError, match="unknown"):
        build_run_config(real_prereg, prof, {})
    prof = Profile("p", False, ["model.*"], {"model.M": "sixteen"})
    with pytest.raises(ConfigError, match="int"):
        build_run_config(real_prereg, prof, {})


def test_batch_size_is_frozen_across_arms(real_prereg):
    """Batch size changes optimization; varying it per M would confound H1."""
    assert "train.batch_size" in real_prereg.frozen
    assert "train.batch_size" not in real_prereg.sweep_keys


def test_editing_the_amendment_log_after_freezing_is_detected(tmp_path):
    import shutil
    from neptune.config import load_prereg
    (tmp_path / "configs").mkdir()
    shutil.copy("configs/preregistered_v1_2_3.yaml", tmp_path / "configs" / "p.yaml")
    shutil.copy("AMENDMENTS.md", tmp_path / "AMENDMENTS.md")
    load_prereg(tmp_path / "configs" / "p.yaml")                       # intact: loads
    with open(tmp_path / "AMENDMENTS.md", "a") as f:
        f.write("\nquietly changed my mind\n")
    with pytest.raises(ConfigError, match="changed after freezing"):
        load_prereg(tmp_path / "configs" / "p.yaml")

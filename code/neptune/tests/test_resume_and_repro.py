"""A run interrupted mid-epoch and resumed must be bit-identical to an
uninterrupted run -- and two uninterrupted runs must be identical too."""
import pandas as pd
import pytest

from neptune.config import build_run_config
from neptune.pipeline.artifacts import load_artifacts
from neptune.training import trainer


class Crash(RuntimeError):
    pass


def _run(cfg, art, rdir, crash_after=None, resume=True):
    calls = {"n": 0}

    def tb(*a, **k):
        if crash_after is not None and calls["n"] >= crash_after:
            raise Crash()
        calls["n"] += 1
        return trainer.neptune_train_batch(*a, **k)
    return trainer.fit_and_evaluate(cfg, art, rdir, "cpu", experiment="F01_BANDWIDTH_PROBE",
                                    train_batch=tb, resume=resume, log=lambda *x: None)


def test_resume_is_bit_identical(data_dir, prereg, profile, tmp_path):
    cfg = build_run_config(prereg, profile, {"model.M": 2, "model.h": 0.3, "model.sigma": 0.25, "seed": 0})
    art = load_artifacts(data_dir)
    a = _run(cfg, art, tmp_path / "a")
    b = _run(cfg, art, tmp_path / "b")
    va = pd.read_parquet(tmp_path / "a" / "peruser_val.parquet")
    vb = pd.read_parquet(tmp_path / "b" / "peruser_val.parquet")
    pd.testing.assert_frame_equal(va, vb)                        # reproducible
    with pytest.raises(Crash):
        _run(cfg, art, tmp_path / "c", crash_after=3)            # ckpt every 2 batches
    assert (tmp_path / "c" / "checkpoint_last.pt").exists()
    assert not (tmp_path / "c" / "summary.json").exists()
    c = _run(cfg, art, tmp_path / "c")
    vc = pd.read_parquet(tmp_path / "c" / "peruser_val.parquet")
    pd.testing.assert_frame_equal(va, vc)                        # resumed == uninterrupted
    assert a["val"]["value"] == c["val"]["value"] == b["val"]["value"]


def test_checkpoint_rng_survives_non_cpu_style_load(tmp_path, cfg):
    """Regression: RNG states are restored as CPU ByteTensors regardless of map_location."""
    import torch
    from conftest import tiny_model
    from neptune.heads.negatives import NegativeSampler
    from neptune.training import checkpoint as ck
    import numpy as np
    model, _ = tiny_model(cfg)
    opt = torch.optim.AdamW(model.parameters())
    samp = NegativeSampler(np.ones(40, bool), np.ones(40), "uniform", "cpu", 0)
    ck.save(tmp_path / "c.pt", model=model, optimizer=opt, scheduler=None, sampler=samp,
            epoch=1, batch_pos=2, step=3)
    a = samp.sample(2, 5)
    ck.load(tmp_path / "c.pt", model=model, optimizer=opt, sampler=samp, map_location="cuda")
    assert torch.equal(samp.sample(2, 5), a)

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
torch.set_num_threads(2)

from neptune.config import build_run_config, load_prereg, load_profile  # noqa: E402

FIX = ROOT / "tests" / "fixtures"


@pytest.fixture(scope="session")
def prereg():
    return load_prereg(FIX / "prereg_smoke.yaml")


@pytest.fixture(scope="session")
def real_prereg():
    return load_prereg(ROOT / "configs" / "preregistered_v1_2_3.yaml")


@pytest.fixture(scope="session")
def profile():
    return load_profile(FIX / "smoke_claim.yaml")


@pytest.fixture(scope="session")
def cfg(prereg, profile):
    return build_run_config(prereg, profile, {"model.M": 4, "model.h": 0.3, "model.sigma": 0.25, "seed": 0})


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory, prereg, profile):
    from neptune.pipeline.covariates import build_covariates
    from neptune.pipeline.prepare import prepare_dataset
    from neptune.synthetic import write_synthetic_ml25m
    root = tmp_path_factory.mktemp("np")
    raw = write_synthetic_ml25m(root / "raw", n_users=260)
    base = build_run_config(prereg, profile, for_training=False)
    prepare_dataset(raw, root / "data", base)
    build_covariates(root / "data", base)
    return root / "data"


def tiny_model(cfg, n_items=40, d_x=12, seed=0):
    from neptune.model import NeptuneH1
    torch.manual_seed(seed)
    X = torch.randn(n_items, d_x)
    return NeptuneH1(cfg, X, seed=seed), X


def write_fake_pilot(out_root, prereg, d17_by_run=None, profile_name="smoke_pilot"):
    """Materialize canonical pilot runs of THIS prereg namespace (with provenance and a
    contract-conforming data manifest) carrying chosen final D17 values."""
    import pandas as pd
    import yaml
    from pathlib import Path
    from neptune.common import sha256_file, write_json_atomic
    from neptune.config import load_profile
    from neptune.logging.registry import run_dir
    from neptune.pipeline.gates import pilot_sweeps
    from neptune.pipeline.prepare import DATA_SCHEMA
    prof = load_profile(FIX / f"{profile_name}.yaml")
    ddir = Path(out_root) / "fake_pilot_data"
    ddir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(ddir / "manifest.json", {"schema": DATA_SCHEMA, "prereg_sha256": prereg.sha256,
                                               "n_users": prereg.gates["open1"]["data_n_users"]})
    for k, sw in enumerate(pilot_sweeps(prereg)):
        v = (d17_by_run or {}).get(k, 0.4)
        rdir = run_dir(out_root, build_run_config(prereg, prof, sw))
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / "config.yaml").write_text(yaml.safe_dump(sw))
        write_json_atomic(rdir / "summary.json", {"prereg_sha256": prereg.sha256,
                                                  "D17_final": {"mean_pairwise_cos": v, "effective_rank": 10.0}})
        write_json_atomic(rdir / "provenance.json", {
            "implementation_sha256": prereg.implementation_sha256, "protocol_critical": True,
            "data_dir": str(ddir), "data_manifest_sha256": sha256_file(ddir / "manifest.json")})
        pd.DataFrame({"metric": "D17_mean_pairwise_cos", "epoch": [0], "value": [v]}).to_parquet(rdir / "metrics.parquet")
    return prof


def keep_gate(out_root, prereg):
    """Freeze a KEEP decision backed by (fake) pilot evidence so claim runs may start."""
    from neptune.pipeline.gates import decide_open1, freeze_open1, gate_path
    prof = write_fake_pilot(out_root, prereg)
    return freeze_open1(decide_open1(prereg, prof, out_root), gate_path(out_root, prereg))

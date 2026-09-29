from neptune.benchmarks.harness import ratchet, run_benchmarks
from neptune.config import build_run_config


def test_benchmark_harness_runs_and_reports_oracle(prereg, profile):
    cfg = build_run_config(prereg, profile, {"model.M": 4}, for_training=False)
    res = run_benchmarks(cfg, prereg.engineering["benchmark"], "cpu")
    for k in ("kernel", "state", "rk4", "rk4_oracle", "train", "train_oracle"):
        assert res[k]["events_per_sec"] > 0 and res[k]["p10_s"] <= res[k]["p90_s"]
    assert 0 < res["train_E_vec"] < 5 and 0 < res["rk4_E_vec"] < 5
    r = ratchet(res, {"train": 10**9}, None, "NVIDIA GeForce RTX 3090")
    assert not r["enforced"] and not r["all_pass"]                 # informational off-reference
    r = ratchet(res, {"train": 1}, "NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 3090")
    assert r["enforced"] and r["all_pass"]

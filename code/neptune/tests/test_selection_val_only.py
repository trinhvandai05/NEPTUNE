import numpy as np
import pandas as pd
import pytest
import yaml

from neptune.pipeline.plans import confirmatory_sweeps, f01_sweeps, f02_seed0_sweeps
from neptune.pipeline.selection import (collect_val_table, decide_sigma_policy, freeze_selection,
                                        select_per_arm)


SHA = "abcdef0123456789"


def _fake_runs(root, values):
    for (M, h, s), v in values.items():
        d = root / "runs" / "p" / SHA[:12] / f"neptune_M{M}_h{h}_sg{s}"
        (d / "sealed").mkdir(parents=True)
        cfg = {"model": {"M": M, "h": h, "sigma": s}, "seed": 0, "prereg_sha256": SHA, "train": {"negatives": "uniform"},
               "data": {"ordinal_time": True}, "baseline": {"kind": "none"}}
        (d / "config.yaml").write_text(yaml.safe_dump(cfg))
        pd.DataFrame({"u": [0, 1], "ndcg@10": [v, v]}).to_parquet(d / "peruser_val.parquet")
        pd.DataFrame({"u": [0, 1], "ndcg@10": [0.5, 0.5]}).to_parquet(d / "sealed" / "peruser_test.parquet")


def test_selection_ignores_sealed_test(tmp_path):
    H, S = [0.2, 0.5], [0.2, 0.4]
    vals = {(M, h, s): 0.1 + 0.01 * M + (0.02 if h == 0.5 else 0) + (0.005 if s == 0.2 else 0)
            for M in (1, 4) for h in H for s in S}
    _fake_runs(tmp_path, vals)
    val = collect_val_table(tmp_path, "p", SHA, metric="ndcg@10")
    pol = decide_sigma_policy(val, [1, 4], S, H, 0.01)
    sel1 = select_per_arm(val, pol, [1, 4], H, S, "ndcg@10")
    for f in tmp_path.rglob("peruser_test.parquet"):              # corrupt every sealed file
        pd.DataFrame({"u": [0, 1], "ndcg@10": [np.random.rand(), 9.0]}).to_parquet(f)
    sel2 = select_per_arm(collect_val_table(tmp_path, "p", SHA, metric="ndcg@10"), pol, [1, 4], H, S, "ndcg@10")
    assert sel1["arms"] == sel2["arms"] and sel1["selection_split"] == "val"
    from pathlib import Path
    assert all(Path(f["file"]).name == "peruser_val.parquet" and Path(f["file"]).parent.name != "sealed"
               for f in sel1["source_files"])
    assert sel1["arms"]["4"]["h"] == 0.5


def test_sigma_policy_is_symmetric_and_thresholded():
    H, S = [0.2], [0.2, 0.4]
    rows = [{"M": 1, "h": 0.2, "sigma": 0.2, "value": 0.10}, {"M": 1, "h": 0.2, "sigma": 0.4, "value": 0.08},
            {"M": 16, "h": 0.2, "sigma": 0.2, "value": 0.20}, {"M": 16, "h": 0.2, "sigma": 0.4, "value": 0.201}]
    pol = decide_sigma_policy(pd.DataFrame(rows), [1, 16], S, H, 0.01)
    assert pol["shared_sigma"] == 0.2              # neutral choice, not the treatment arm's argmax
    assert pol["mode"] == "shared" and max(pol["cost_by_arm"].values()) < 0.01
    rows[2]["value"] = 0.15
    assert decide_sigma_policy(pd.DataFrame(rows), [1, 16], S, H, 0.01)["mode"] == "per_arm"


def test_freeze_is_write_once(tmp_path):
    sel = {"arms": {"1": {"h": 0.2}}, "selection_split": "val"}
    freeze_selection(sel, tmp_path / "s.json")
    freeze_selection(sel, tmp_path / "s.json")                     # identical: no-op
    with pytest.raises(RuntimeError, match="refusing"):
        freeze_selection({"arms": {"1": {"h": 0.5}}, "selection_split": "val"}, tmp_path / "s.json")


def test_run_counts_match_the_plan(real_prereg):
    shared = {"mode": "shared", "shared_sigma": 0.30}
    per = {"mode": "per_arm", "shared_sigma": 0.30}
    f01 = {tuple(sorted(s.items())) for s in f01_sweeps(real_prereg)}
    extra_shared = {tuple(sorted(s.items())) for s in f02_seed0_sweeps(real_prereg, shared)} - f01
    extra_per = {tuple(sorted(s.items())) for s in f02_seed0_sweeps(real_prereg, per)} - f01
    sel = {"arms": {str(M): {"h": 0.35, "sigma": 0.3} for M in real_prereg.grids["M"]}}
    conf = confirmatory_sweeps(real_prereg, sel)
    assert len(f01) == 18 and len(conf) == 18                  # 2 primary arms x 5 + 4 arms x 2
    assert all(s["seed"] != 0 for s in conf)                  # tuning seed never confirmatory
    assert len(f01) + len(extra_shared) + len(conf) == 48
    assert len(f01) + len(extra_per) + len(conf) == 72

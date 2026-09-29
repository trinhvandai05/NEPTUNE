"""Hyperparameter selection -- validation data ONLY.

Everything here reads `peruser_val.parquet` and `config.yaml` and nothing else.
The sealed test files live in `sealed/` and no function in this module opens
that directory; tests/test_selection_val_only.py corrupts them and checks the
selection is unchanged.

sigma policy (amendment A2, made symmetric here)
------------------------------------------------
For every F01 arm M, b_M(sigma) = best validation metric over h at that sigma.
The shared sigma is the one maximizing the sum over arms of b_M(sigma)/max b_M
(a neutral choice: neither arm's optimum is privileged).  The cost to arm M of
sharing is  1 - b_M(sigma_shared) / max_sigma b_M(sigma).  If ANY arm's cost
exceeds the preregistered threshold, sigma is tuned per arm (3x3 grid per arm);
otherwise every arm gets 3 h-trials at the shared sigma.  Equal search budget
per arm either way.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yaml

from ..common import read_json, sha256_file, write_json_atomic


def collect_val_table(out_root, profile: str, prereg_sha256: str, *, metric: str,
                      negatives: str = "uniform", ordinal: bool = True, seeds=(0,)) -> pd.DataFrame:
    """Only runs of THIS preregistration version (namespace <profile>/<sha12>)."""
    from ..logging.registry import namespace
    rows = []
    base = Path(out_root) / "runs" / namespace(profile, prereg_sha256)
    if not base.exists():
        return pd.DataFrame(columns=["M", "h", "sigma", "seed", "value", "n_users", "file"])
    for rdir in sorted(base.iterdir()):
        cfg_p, val_p = rdir / "config.yaml", rdir / "peruser_val.parquet"
        if not (cfg_p.exists() and val_p.exists()):
            continue
        cfg = yaml.safe_load(cfg_p.read_text())
        if cfg["prereg_sha256"] != prereg_sha256:
            raise RuntimeError(f"{rdir} carries a different preregistration hash")
        if cfg["baseline"]["kind"] != "none" or cfg["train"]["negatives"] != negatives:
            continue
        if cfg["data"]["ordinal_time"] != ordinal or cfg["seed"] not in seeds:
            continue
        val = pd.read_parquet(val_p)
        rows.append({"M": int(cfg["model"]["M"]), "h": float(cfg["model"]["h"]),
                     "sigma": float(cfg["model"]["sigma"]), "seed": int(cfg["seed"]),
                     "value": float(val[metric].mean()), "n_users": int(len(val)),
                     "file": str(val_p), "sha256": sha256_file(val_p)})
    return pd.DataFrame(rows)


def decide_sigma_policy(val: pd.DataFrame, f01_M, sigma_grid, h_grid, threshold: float) -> dict:
    best_by_sigma, best = {}, {}
    for M in f01_M:
        sub = val[val["M"] == M]
        b = {}
        for s in sigma_grid:
            cell = sub[(sub["sigma"] - s).abs() < 1e-9]
            if len(cell) < len(h_grid):
                raise RuntimeError(f"F01 incomplete: M={M}, sigma={s} has {len(cell)}/{len(h_grid)} h-runs")
            b[float(s)] = float(cell["value"].max())
        best_by_sigma[int(M)] = b
        best[int(M)] = max(b.values())

    def norm_sum(s):
        return sum(best_by_sigma[M][s] / best[M] for M in best_by_sigma)
    shared = max(sorted(best_by_sigma[int(f01_M[0])]), key=norm_sum)
    costs = {M: 1.0 - best_by_sigma[M][shared] / best[M] for M in best_by_sigma}
    mode = "per_arm" if max(costs.values()) > threshold else "shared"
    return {"mode": mode, "shared_sigma": float(shared), "cost_by_arm": costs,
            "threshold": float(threshold), "best_by_sigma": best_by_sigma,
            "best_sigma_by_arm": {M: max(b, key=b.get) for M, b in best_by_sigma.items()},
            "selection_split": "val"}


def candidate_grid(policy: dict, h_grid, sigma_grid):
    sigmas = [policy["shared_sigma"]] if policy["mode"] == "shared" else list(sigma_grid)
    return [(float(h), float(s)) for s in sigmas for h in h_grid]


def select_per_arm(val: pd.DataFrame, policy: dict, M_grid, h_grid, sigma_grid, metric: str) -> dict:
    cands = candidate_grid(policy, h_grid, sigma_grid)
    arms, files = {}, []
    for M in M_grid:
        sub = val[val["M"] == M]
        table = []
        for h, s in cands:
            cell = sub[((sub["h"] - h).abs() < 1e-9) & ((sub["sigma"] - s).abs() < 1e-9)]
            if cell.empty:
                raise RuntimeError(f"missing seed-0 tuning run for M={M}, h={h}, sigma={s}")
            r = cell.iloc[0]
            table.append({"h": h, "sigma": s, "value": float(r["value"])})
            files.append({"file": r["file"], "sha256": r["sha256"]})
        # deterministic tie-break: higher value, then smaller h, then smaller sigma
        table.sort(key=lambda c: (-c["value"], c["h"], c["sigma"]))
        arms[str(int(M))] = {"h": table[0]["h"], "sigma": table[0]["sigma"],
                             "val_value": table[0]["value"], "candidates": table}
    return {"selection_split": "val", "metric": metric, "policy": policy["mode"],
            "shared_sigma": policy["shared_sigma"], "arms": arms, "source_files": files}


def freeze_selection(selection: dict, path) -> dict:
    """Write once.  Re-running with the same result is a no-op; a DIFFERENT
    result is refused -- h*(M) cannot be quietly re-chosen after test is unsealed."""
    path = Path(path)
    if path.exists():
        old = read_json(path)
        if old["arms"] != selection["arms"]:
            raise RuntimeError(f"{path} already frozen with a different selection; refusing to overwrite")
        return old
    selection = dict(selection, frozen_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    write_json_atomic(path, selection)
    return selection


# ------------------------------------------------ provenance of selection artifacts

def _stamp(obj: dict, prereg) -> dict:
    from ..provenance import check_implementation
    return dict(obj, prereg_sha256=prereg.sha256,
                implementation_sha256=check_implementation(prereg, strict=True))


def policy_source_files(val: pd.DataFrame, f01_M) -> list:
    rows = val[val["M"].isin(f01_M)].sort_values("file")
    return [{"file": r["file"], "sha256": r["sha256"]} for _, r in rows.iterrows()]


def freeze_policy(policy: dict, val: pd.DataFrame, f01_M, prereg, path) -> dict:
    """sigma_policy.json: write-once, stamped with prereg + (frozen) code and the hashes
    of every F01 validation file it was computed from."""
    path = Path(path)
    policy = _stamp(dict(policy, source_files=policy_source_files(val, f01_M)), prereg)
    if path.exists():
        old = read_json(path)
        if (old["mode"], old["shared_sigma"]) != (policy["mode"], policy["shared_sigma"]):
            raise RuntimeError(f"{path} already frozen with a different policy; refusing to overwrite")
        return old
    write_json_atomic(path, policy)
    return policy


def anchor_selection(selection: dict, prereg, data_dir) -> dict:
    """Stamp prereg + code, and ANCHOR the data artifact and the H1 covariates: they
    must exist before h*(M) is frozen, and can never change afterwards (checked
    again before the test set is unsealed)."""
    data_dir = Path(data_dir)
    sem = data_dir / "semantics_manifest.json"
    if not sem.exists():
        raise RuntimeError("build the H1 covariates before freezing the selection")
    return _stamp(dict(selection, data_manifest_sha256=sha256_file(data_dir / "manifest.json"),
                       semantics_manifest_sha256=sha256_file(sem)), prereg)


def verify_frozen_selection(prereg, out_root, profile_name: str, data_dir) -> dict:
    """Before confirmatory runs and before unsealing test: the frozen policy and
    selection belong to this prereg and code, every validation file they cite is
    byte-identical, the data/covariate anchors still hold, and RE-DERIVING the policy
    and h*(M) from the validation runs with the frozen rules gives the same answer."""
    from ..logging.registry import selection_dir
    from ..provenance import expected_implementation
    sdir = selection_dir(out_root, profile_name, prereg.sha256)
    pol, sel = read_json(sdir / "sigma_policy.json"), read_json(sdir / "selected_h.json")
    impl = expected_implementation(prereg)
    for name, obj in (("sigma_policy", pol), ("selected_h", sel)):
        if obj.get("prereg_sha256") != prereg.sha256 or obj.get("implementation_sha256") != impl:
            raise RuntimeError(f"{name}.json was produced under another preregistration or code")
        for src in obj["source_files"]:
            f = Path(src["file"])
            if not f.exists() or sha256_file(f) != src["sha256"]:
                raise RuntimeError(f"validation file cited by {name}.json changed or vanished: {f}")
    data_dir = Path(data_dir)
    if (sha256_file(data_dir / "manifest.json") != sel["data_manifest_sha256"]
            or sha256_file(data_dir / "semantics_manifest.json") != sel["semantics_manifest_sha256"]):
        raise RuntimeError("data artifact or H1 covariates changed after h*(M) was frozen")
    g, tun = prereg.grids, prereg.tuning
    val = collect_val_table(out_root, profile_name, prereg.sha256, metric=tun["selection_metric"],
                            negatives=g["negatives_primary"], seeds=tuple(g["seeds_tuning"]))
    pol2 = decide_sigma_policy(val[val["M"].isin(g["f01_M"])], g["f01_M"], g["sigma"], g["h"],
                               tun["sigma_cost_threshold"])
    if (pol2["mode"], pol2["shared_sigma"]) != (pol["mode"], pol["shared_sigma"]):
        raise RuntimeError("sigma_policy.json does not match the policy re-derived from validation")
    sel2 = select_per_arm(val, pol, g["M"], g["h"], g["sigma"], tun["selection_metric"])
    if sel2["arms"] != sel["arms"]:
        raise RuntimeError("selected_h.json does not match h*(M) re-derived from validation")
    return sel

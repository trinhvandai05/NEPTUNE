"""Hard gates that must pass before ANY claim-eligible run starts.

OPEN-1 (AMENDMENTS.md).  A claim-INELIGIBLE pilot on ML-25M -- its exact
configuration is the frozen `gates.open1` contract, checked key by key against
the claim configuration -- decides whether this preregistration is kept or
amendment v1.3 (catalog standardization of encoder outputs) must be written
first.  The decision is a write-once JSON artifact that records the prereg hash
and the SHA-256 of every pilot summary it was based on.  `require_open1()` is
called by the runner for every claim-eligible run: no decision, a stale
decision, a decision based on altered pilot files, or an ADOPT verdict all stop
the run.  The D17 threshold is the single preregistered value
acceptance.d17_max_mean_cos, shared with acceptance condition C6.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from ..common import read_json, sha256_file, write_json_atomic
from ..config import build_run_config
from ..logging.registry import run_dir

KEEP, ADOPT = "KEEP_CURRENT_PREREG", "ADOPT_V1_3"


class GateError(RuntimeError):
    """A preregistered gate has not been passed."""


# Keys a pilot config may legitimately differ on from the claim configuration.
# data.n_users and train.warmup_steps must then equal the frozen pilot contract.
_PILOT_FREE = {"profile", "claim_eligible", "train.ckpt_every_batches"}
_PILOT_CONTRACT = {"data.n_users": "data_n_users", "train.warmup_steps": "train_warmup_steps"}


def _contract(prereg) -> dict:
    c = (prereg.gates or {}).get("open1")
    if not c:
        raise GateError("preregistration has no gates.open1 pilot contract")
    return c


def pilot_sweeps(prereg) -> list:
    """The canonical pilot grid, read from the frozen contract (not from a profile)."""
    c = _contract(prereg)
    return [{"model.M": int(M), "model.h": float(h), "model.sigma": float(c["sigma"]), "seed": int(c["seed"])}
            for M in c["M"] for h in c["h"]]


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def check_pilot_config(prereg, cfg) -> None:
    """The pilot config must be the claim config for that sweep, except for the
    contract keys (which must equal the frozen contract values) and bookkeeping."""
    from ..config import Profile
    c = _contract(prereg)
    if cfg.claim_eligible:
        raise GateError("the OPEN-1 pilot must run under a claim-INELIGIBLE profile")
    sweep = {"model.M": cfg.model.M, "model.h": cfg.model.h, "model.sigma": cfg.model.sigma, "seed": cfg.seed}
    base = build_run_config(prereg, Profile("__claim_base__", False, [], {}), sweep).to_dict()
    mine = cfg.to_dict()
    fb, fm = _flat(base), _flat(mine)
    bad = {k: (fb.get(k), fm.get(k)) for k in set(fb) | set(fm)
           if k not in _PILOT_FREE and k not in _PILOT_CONTRACT and fb.get(k) != fm.get(k)}
    for key, ckey in _PILOT_CONTRACT.items():
        if fm[key] != c[ckey]:
            bad[key] = (c[ckey], fm[key])
    if bad:
        raise GateError(f"non-canonical OPEN-1 pilot (expected contract vs got): {bad}")


def decide_open1(prereg, pilot_profile, out_root) -> dict:
    """Reads ONLY the canonical pilot runs of this preregistration's namespace and
    verifies each one's configuration, code and data provenance."""
    from ..provenance import check_implementation
    from .prepare import DATA_SCHEMA
    check_implementation(prereg, strict=True)                  # the deciding code is frozen code
    thr = float(prereg.acceptance["d17_max_mean_cos"])
    c = _contract(prereg)
    runs = []
    for sw in pilot_sweeps(prereg):
        cfg = build_run_config(prereg, pilot_profile, sw)
        check_pilot_config(prereg, cfg)
        rdir = run_dir(out_root, cfg)
        summ_p, prov_p = rdir / "summary.json", rdir / "provenance.json"
        if not summ_p.exists() or not prov_p.exists():
            raise GateError(f"pilot run missing or unfinished: {rdir}")
        summ, prov = read_json(summ_p), read_json(prov_p)
        if summ.get("prereg_sha256") != prereg.sha256:
            raise GateError(f"{rdir} was produced under another preregistration")
        if not prov.get("protocol_critical"):
            raise GateError(f"{rdir} was not run as a protocol-critical P00_PILOT run")
        if prereg.implementation_sha256 and prov.get("implementation_sha256") != prereg.implementation_sha256:
            raise GateError(f"{rdir} was produced by code other than the frozen implementation")
        man_p = Path(prov["data_dir"]) / "manifest.json"
        if not man_p.exists() or sha256_file(man_p) != prov["data_manifest_sha256"]:
            raise GateError(f"data artifact of {rdir} is missing or changed since the run")
        man = read_json(man_p)
        if (man.get("schema") != DATA_SCHEMA or man.get("prereg_sha256") != prereg.sha256
                or int(man.get("n_users", -1)) != int(c["data_n_users"])):
            raise GateError(f"pilot data artifact {man_p} does not match the pilot contract")
        m = pd.read_parquet(rdir / "metrics.parquet")
        traj = m[m["metric"] == "D17_mean_pairwise_cos"].sort_values("epoch")["value"].tolist()
        d17 = summ["D17_final"]
        runs.append({"run": rdir.name, "M": sw["model.M"], "h": sw["model.h"],
                     "D17_trajectory": [round(v, 5) for v in traj],
                     "D17_final": float(d17["mean_pairwise_cos"]),
                     "D17_final_descriptive": {k: v for k, v in d17.items() if k != "mean_pairwise_cos"},
                     "evidence": {name: {"path": str(f), "sha256": sha256_file(f)}
                                  for name, f in (("summary", summ_p), ("config", rdir / "config.yaml"),
                                                  ("provenance", prov_p), ("data_manifest", man_p))}})
    by_h = {}
    for r in runs:
        by_h.setdefault(r["h"], {})[r["M"]] = r["D17_final"]
    diff = {str(h): max(v.values()) - min(v.values()) for h, v in by_h.items()}
    worst = max(r["D17_final"] for r in runs)
    return {"gate": "OPEN-1", "prereg_sha256": prereg.sha256, "prereg_version": prereg.version,
            "implementation_sha256": prereg.implementation_sha256, "contract": c,
            "pilot_profile": pilot_profile.name, "threshold": thr, "max_D17_final": worst,
            "differential_D17_across_M": diff, "runs": runs,
            "verdict": ADOPT if worst > thr else KEEP}


def gate_path(out_root, prereg) -> Path:
    return Path(out_root) / "gates" / prereg.sha12 / "open1_decision.json"


def freeze_open1(decision: dict, path) -> dict:
    path = Path(path)
    if path.exists():
        old = read_json(path)
        same = (old["verdict"] == decision["verdict"]
                and [r["evidence"] for r in old["runs"]] == [r["evidence"] for r in decision["runs"]])
        if not same:
            raise GateError(f"{path} already frozen with a different decision; refusing to overwrite")
        return old
    decision = dict(decision, frozen_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    write_json_atomic(path, decision)
    return decision


def require_open1(prereg, out_root) -> dict | None:
    if not prereg.gates.get("open1_required", False):
        return None
    p = gate_path(out_root, prereg)
    if not p.exists():
        raise GateError("OPEN-1 has not been decided for this preregistration. Run scripts/run_pilot.py "
                        "then scripts/pilot_report.py before any claim-eligible run.")
    d = read_json(p)
    if d.get("prereg_sha256") != prereg.sha256:
        raise GateError(f"{p} belongs to another preregistration")
    if d.get("verdict") != KEEP:
        raise GateError("OPEN-1 decided ADOPT_V1_3: write amendment v1.3 (catalog standardization) and "
                        "re-freeze. Claim runs under this preregistration are blocked.")
    if not d.get("runs"):
        raise GateError(f"{p} lists no pilot runs")
    if prereg.implementation_sha256 and d.get("implementation_sha256") != prereg.implementation_sha256:
        raise GateError(f"{p} was decided under a different frozen implementation")
    d17s = []
    for r in d["runs"]:
        for name, ev in r["evidence"].items():                # summary, config, provenance, data manifest
            f = Path(ev["path"])
            if not f.exists() or sha256_file(f) != ev["sha256"]:
                raise GateError(f"pilot evidence changed after the OPEN-1 decision ({name}): {f}")
        d17s.append(float(read_json(r["evidence"]["summary"]["path"])["D17_final"]["mean_pairwise_cos"]))
    # The decision file is not trusted: re-derive the verdict from the verified evidence.
    thr = float(prereg.acceptance["d17_max_mean_cos"])
    worst = max(d17s)
    expected = ADOPT if worst > thr else KEEP
    if (d["verdict"] != expected or float(d["threshold"]) != thr
            or float(d["max_D17_final"]) != worst or len(d17s) != len(pilot_sweeps(prereg))):
        raise GateError(f"{p} is inconsistent with its own evidence: recorded verdict {d['verdict']} "
                        f"(max D17 {d['max_D17_final']}, threshold {d['threshold']}) vs recomputed "
                        f"{expected} (max D17 {worst}, threshold {thr})")
    return d

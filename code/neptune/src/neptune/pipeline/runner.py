"""Execute one run (with guarded reuse) or a list of runs.

Protocol-critical runs are claim-eligible runs AND the OPEN-1 pilot (P00_PILOT):
the pilot is claim-ineligible, but its outcome unlocks the claim runs, so it gets
the same strictness -- exact artifact match, frozen implementation fingerprint,
and guarded reuse.  Every run writes provenance.json (implementation fingerprint,
git state, data manifest hash) before training starts.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from ..common import read_json, sha256_file, write_json_atomic
from ..config import ConfigError, build_run_config
from ..logging.manifest import git_info, write_run_manifests
from ..logging.registry import EXPERIMENTS, run_dir
from ..provenance import check_implementation
from .artifacts import check_artifacts_match, load_artifacts, verify_data_payload
from .gates import check_pilot_config, require_open1

PROTOCOL_EXPERIMENTS = {"P00_PILOT"}
_ART_CACHE: dict = {}


def _artifacts(data_dir):
    key = str(Path(data_dir).resolve())
    if key not in _ART_CACHE:
        _ART_CACHE[key] = load_artifacts(data_dir)
    return _ART_CACHE[key]


def _note_experiment(rdir: Path, experiment: str):
    p = rdir / "experiments.json"
    used = read_json(p) if p.exists() else []
    if experiment not in used:
        write_json_atomic(p, used + [experiment])


def execute(prereg, profile, sweep: dict, *, data_dir, out_root, device, experiment: str,
            resume: bool = True, log=print) -> dict:
    if experiment not in EXPERIMENTS:
        raise KeyError(f"unregistered experiment '{experiment}'")
    cfg = build_run_config(prereg, profile, sweep)
    strict = cfg.claim_eligible or experiment in PROTOCOL_EXPERIMENTS
    if cfg.claim_eligible:
        require_open1(prereg, out_root)                       # hard gate: raises GateError
    if experiment == "P00_PILOT":
        check_pilot_config(prereg, cfg)                       # a non-canonical pilot cannot even start
    impl = check_implementation(prereg, strict)
    if strict:
        verify_data_payload(data_dir, prereg, strict=False)   # payload bytes == recorded hashes
    data_manifest_sha = sha256_file(Path(data_dir) / "manifest.json")
    rdir = run_dir(out_root, cfg)

    if (rdir / "summary.json").exists():
        stored = yaml.safe_load((rdir / "config.yaml").read_text())
        want = yaml.safe_load(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
        if stored != want:
            diff = sorted(k for k in set(stored) | set(want) if stored.get(k) != want.get(k))
            raise ConfigError(f"refusing to reuse {rdir}: stored config differs from the "
                              f"requested one in {diff}")
        if read_json(rdir / "summary.json").get("prereg_sha256") != cfg.prereg_sha256:
            raise ConfigError(f"refusing to reuse {rdir}: preregistration hash differs")
        prov = read_json(rdir / "provenance.json") if (rdir / "provenance.json").exists() else {}
        if strict and prov.get("implementation_sha256") != impl:
            raise ConfigError(f"refusing to reuse {rdir}: it was produced by different code "
                              f"({str(prov.get('implementation_sha256'))[:12]} != {impl[:12]})")
        if strict and prov.get("data_manifest_sha256") != data_manifest_sha:
            raise ConfigError(f"refusing to reuse {rdir}: it was trained on a different data artifact")
        _note_experiment(rdir, experiment)
        log(f"[run] reuse {rdir.name} (already finished, provenance verified)")
        return read_json(rdir / "summary.json")

    art = _artifacts(data_dir)
    diff = check_artifacts_match(art, cfg, strict=strict,
                                 implementation_sha256=prereg.implementation_sha256 if strict else None)
    if diff:
        log(f"[run] WARNING exploratory run on a mismatching artifact: {diff}")
    write_run_manifests(rdir, cfg, device)
    write_json_atomic(rdir / "provenance.json", {
        "implementation_sha256": impl, "implementation_frozen": prereg.implementation_sha256,
        "git": git_info(), "data_dir": str(Path(data_dir).resolve()),
        "data_manifest_sha256": data_manifest_sha, "protocol_critical": strict,
        "experiment": experiment})
    _note_experiment(rdir, experiment)
    if cfg.baseline.kind == "none":
        from ..training.trainer import fit_and_evaluate
        summary = fit_and_evaluate(cfg, art, rdir, device, experiment=experiment, resume=resume, log=log)
    else:
        from ..baselines.sasrec import run_baseline
        summary = run_baseline(cfg, art, rdir, device, experiment=experiment, resume=resume, log=log)
    log(f"[run] {rdir.name}: val {summary['val']['metric']} = {summary['val']['value']:.4f}")
    return summary


def execute_many(prereg, profile, sweeps, **kw) -> list:
    return [execute(prereg, profile, s, **kw) for s in sweeps]

"""H1_ANALYSIS: unseal test results for the frozen h*(M) only, run D0-D2, verdict."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yaml

from ..common import read_json, sha256_file, write_json_atomic
from ..config import build_run_config
from ..evaluation.h1_analysis import primary_contrast, run_h1_analysis
from ..evaluation.report import render_markdown
from ..evaluation.stop_rules import evaluate_acceptance
from ..logging.registry import report_dir, run_dir, selection_dir
from ..provenance import check_implementation
from .artifacts import verify_data_payload, verify_semantics
from .covariates import SEMANTICS_SCHEMA
from .selection import verify_frozen_selection
from .gates import GateError, require_open1
from .prepare import DATA_SCHEMA


def implementation_consistent(impls: set, frozen: str | None) -> bool:
    """All analysed runs share ONE recorded implementation fingerprint, equal to the
    frozen one when a freeze exists.  Runs from two code versions never mix."""
    return len(impls) == 1 and None not in impls and (frozen is None or impls == {frozen})


def _load(prereg, profile, out_root, sel, M, seed, extra=None):
    arm = sel["arms"][str(int(M))]
    sweep = {"model.M": int(M), "model.h": arm["h"], "model.sigma": arm["sigma"], "seed": int(seed),
             "train.negatives": prereg.grids["negatives_primary"], **(extra or {})}
    rdir = run_dir(out_root, build_run_config(prereg, profile, sweep))
    f = rdir / "sealed" / "peruser_test.parquet"
    if not f.exists():
        return None, rdir, None
    df = pd.read_parquet(f)
    df["M"], df["seed"] = int(M), int(seed)
    return df, rdir, read_json(rdir / "summary.json")


def _robustness(prereg, profile, out_root, sel, extra, metric, alpha) -> dict:
    """Secondary arms at the frozen h*(M): primary contrast only, two-level, never gating."""
    acc = prereg.acceptance
    seeds = prereg.seeds_confirmatory(int(acc["treat_M"]))
    frames = []
    for M in (int(acc["treat_M"]), int(acc["ctrl_M"])):
        for s in seeds:
            df, _, _ = _load(prereg, profile, out_root, sel, M, s, extra)
            if df is None:
                return {"available": False}
            frames.append(df)
    _, c1 = primary_contrast(pd.concat(frames), metric, int(acc["treat_M"]), int(acc["ctrl_M"]), seeds, alpha)
    return {"available": True, **c1}


def analyze_h1(prereg, profile, *, data_dir, out_root) -> dict:
    """Everything is verified BEFORE a single sealed test file is opened: the code
    computing C1-C6 is the frozen code, the data payload and the H1 covariates are
    byte-identical to what was built and anchored, and h*(M) re-derives exactly."""
    out_root = Path(out_root)
    check_implementation(prereg, strict=True)
    base = build_run_config(prereg, profile, for_training=False)
    verify_data_payload(data_dir, prereg, strict=True)
    verify_semantics(data_dir, prereg, asdict(base.semantics))
    sel = verify_frozen_selection(prereg, out_root, profile.name, data_dir)
    sdir = selection_dir(out_root, profile.name, prereg.sha256)
    pol_p = sdir / "sigma_policy.json"
    policy = read_json(pol_p) if pol_p.exists() else None
    g, acc_cfg = prereg.grids, prereg.acceptance
    metric = prereg.tuning["selection_metric"]
    alpha = float(acc_cfg["alpha"])
    tuning = list(g["seeds_tuning"])
    seeds_by_arm = {int(M): prereg.seeds_confirmatory(M) for M in g["M"]}

    frames, claim_flags, d17, missing, impls, data_shas = [], [], [], [], set(), set()
    for M in g["M"]:
        for seed in tuning + seeds_by_arm[int(M)]:
            df, rdir, summ = _load(prereg, profile, out_root, sel, M, seed)
            if df is None:
                missing.append(str(rdir))
                continue
            stored = yaml.safe_load((rdir / "config.yaml").read_text())
            claim_flags.append(bool(stored["claim_eligible"]))
            prov = read_json(rdir / "provenance.json") if (rdir / "provenance.json").exists() else {}
            impls.add(prov.get("implementation_sha256"))
            data_shas.add(prov.get("data_manifest_sha256"))
            d17.append(float(summ["D17_final"]["mean_pairwise_cos"]))
            frames.append(df)
    if missing:
        raise FileNotFoundError("runs missing for the frozen selection:\n  " + "\n  ".join(missing))

    cov = pd.read_parquet(Path(data_dir) / "covariates.parquet")
    sem_man = read_json(Path(data_dir) / "semantics_manifest.json")
    data_man = read_json(Path(data_dir) / "manifest.json")
    sem = sem_man["semantics_cfg"]
    try:
        require_open1(prereg, out_root)
        gate_ok = True
    except GateError:
        gate_ok = False
    an = run_h1_analysis(pd.concat(frames, ignore_index=True), cov, metric=metric,
                         seeds_by_arm=seeds_by_arm, tuning_seeds=tuning,
                         treat_M=int(acc_cfg["treat_M"]), ctrl_M=int(acc_cfg["ctrl_M"]),
                         alpha=alpha, r_min=float(acc_cfg["c5_r_min"]),
                         n_boot_iv=int(acc_cfg["n_boot_iv"]))
    an["R1_logdt"] = _robustness(prereg, profile, out_root, sel, {"data.ordinal_time": False}, metric, alpha)
    other = [n for n in g["negatives"] if n != g["negatives_primary"]]
    an["R2_popularity_negatives"] = (_robustness(prereg, profile, out_root, sel, {"train.negatives": other[0]},
                                                 metric, alpha) if other else {"available": False})
    an["D17_max_mean_cos"] = max(d17)

    procedural = {
        "selection_on_validation": sel.get("selection_split") == "val",
        "per_arm_selection": all(str(int(M)) in sel["arms"] for M in g["M"]),
        "sigma_policy_applied": policy is not None and policy.get("mode") == sel.get("policy"),
        "all_runs_claim_eligible": bool(claim_flags) and all(claim_flags),
        "tuning_seed_excluded": not (set(tuning) & set(an["confirmatory_seeds"])),
        "covariates_verified": (sem == asdict(base.semantics) and sem_man.get("schema") == SEMANTICS_SCHEMA
                                and sem_man.get("prereg_sha256") == prereg.sha256),   # (verify_semantics raised otherwise)
        "selection_verified": True,                                                  # (verify_frozen_selection raised otherwise)
        "data_artifact_matches_prereg": (data_man.get("schema") == DATA_SCHEMA
                                         and data_man.get("prereg_sha256") == prereg.sha256),
        "open1_gate_passed": gate_ok,
        # every analysed run: the SAME code (= the frozen fingerprint when one is frozen) ...
        "single_implementation": implementation_consistent(impls, prereg.implementation_sha256),
        # ... and the SAME data artifact, which is the one being analysed
        "single_data_artifact": data_shas == {sha256_file(Path(data_dir) / "manifest.json")},
        "D17_mean_cos_ok": max(d17) <= float(acc_cfg["d17_max_mean_cos"]),
    }
    acc = evaluate_acceptance(an, acc_cfg, procedural)
    meta = {"profile": profile.name, "prereg_version": prereg.version, "prereg_sha256": prereg.sha256,
            "selection_split": sel.get("selection_split"),
            "sigma_policy": policy.get("mode") if policy else None}
    report = {"meta": meta, "analysis": an, "acceptance": acc}
    rep = report_dir(out_root, profile.name, prereg.sha256)
    write_json_atomic(rep / "h1_report.json", report)
    (rep / "h1_report.md").write_text(render_markdown(an, acc, meta))
    return report

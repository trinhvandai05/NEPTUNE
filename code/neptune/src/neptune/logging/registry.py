"""Experiment registry.  Names carry scientific meaning, not run_013_final2."""

from __future__ import annotations

from pathlib import Path

EXPERIMENTS = {
    "P00_PILOT": "claim-INELIGIBLE engineering pilot: VRAM, throughput, D17 trend (OPEN-1)",
    "I00_DATA": "ML-25M ingest, temporal split, evaluation cohort",
    "I01_COVARIATES": "k=200 raw-content clusters, lambda_hat, tau^2, split-half, tail",
    "F01_BANDWIDTH_PROBE": "M in {1,16} x h x sigma, seed 0, validation only",
    "F02_SEED0_TUNING": "remaining M arms, seed 0, validation selection of h*(M)",
    "F02_CONFIRMATORY": "confirmatory seeds at frozen h*(M) (1-5 for the primary arms, 1-2 otherwise); no re-selection",
    "H1_ROBUSTNESS_LOGDT": "M in {1,16} at h*(M) with log1p(clipped days) time -- NOT raw Delta_t",
    "H1_ROBUSTNESS_POPNEG": "M in {1,16} at h*(M) with popularity-sampled negatives",
    "H1_ANALYSIS": "D1, D2-A..D, acceptance conditions 1-6",
    "B01_SASREC": "SASRec control, 3-trial dropout grid",
    "B02_SASREC_PSI": "SASRec + Psi control, 3-trial sigma grid",
    "BENCH": "throughput ratchet + oracle",
}


def run_key(cfg) -> str:
    """Config-derived key, so an F02 run identical to an F01 run is reused
    instead of recomputed."""
    time_tag = "ord" if cfg.data.ordinal_time else "logdt"
    if cfg.baseline.kind != "none":
        return (f"{cfg.baseline.kind}_do{cfg.baseline.dropout:.2f}_sg{cfg.model.sigma:.2f}"
                f"_{cfg.train.negatives}_{time_tag}_s{cfg.seed}")
    return (f"neptune_M{cfg.model.M}_h{cfg.model.h:.2f}_sg{cfg.model.sigma:.2f}"
            f"_{cfg.train.negatives}_{time_tag}_s{cfg.seed}")


def namespace(profile: str, prereg_sha256: str) -> Path:
    """<profile>/<prereg sha12>: a run made under another preregistration version
    lives in another directory and can never be picked up by this one."""
    return Path(profile) / prereg_sha256[:12]


def run_dir(out_root, cfg) -> Path:
    return Path(out_root) / "runs" / namespace(cfg.profile, cfg.prereg_sha256) / run_key(cfg)


def selection_dir(out_root, profile: str, prereg_sha256: str) -> Path:
    return Path(out_root) / "selection" / namespace(profile, prereg_sha256)


def report_dir(out_root, profile: str, prereg_sha256: str) -> Path:
    return Path(out_root) / "reports" / namespace(profile, prereg_sha256)

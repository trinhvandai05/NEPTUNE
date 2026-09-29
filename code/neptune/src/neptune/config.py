"""Configuration: schema, preregistration, profiles.

Three layers are applied in a fixed order, each with its own permission set:

  1. preregistration `frozen`  -- every value a claim depends on
  2. profile `overrides`       -- only keys the profile lists in `may_override`;
                                  a profile that overrides anything can never be
                                  claim-eligible
  3. sweep                     -- only keys listed in the preregistration's
                                  `sweep_keys` (M, h, sigma, seed, ...)

Anything else raises ConfigError.  The stop rules are therefore protected by
the loader, not by discipline: a claim-eligible run physically cannot be
launched with a warmup, epoch count or eligibility rule that differs from the
preregistered one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from .common import sha256_bytes


class ConfigError(ValueError):
    """Any attempt to build a configuration the preregistration forbids."""


# --------------------------------------------------------------------- schema

@dataclass
class DataCfg:
    positive_threshold: float = 4.0
    n_users: int = 25000
    min_train_events: int = 20
    min_test_events: int = 3
    train_frac: float = 0.85
    val_frac: float = 0.05
    session_gap_seconds: float = 1800.0
    genome_components: int = 256
    cold_min_items: int = 300
    # True  -> Delta_t == 1 (H1 primary: ML timestamps are rating-log times).
    # False -> Delta_t = log1p(clip(days, 0, clip_days)): a LOG-COMPRESSED elapsed
    #          time.  It is NOT raw tau_t - tau_{t-1} and must never be reported
    #          as a continuous-time (H2) result.
    ordinal_time: bool = True
    clip_days: float = 365.0


# Keys that define the dataset artifact itself.  A run whose config disagrees
# with the artifact on any of these is not measuring what it claims to.
ARTIFACT_KEYS = ("positive_threshold", "n_users", "min_train_events",
                 "min_test_events", "train_frac", "val_frac",
                 "session_gap_seconds", "genome_components", "cold_min_items")


@dataclass
class SemanticsCfg:
    n_clusters: int = 200
    kmeans_seed: int = 0
    head_fraction: float = 0.20
    n_boot_reliability: int = 200
    n_strata: int = 5


@dataclass
class ModelCfg:
    M: int = 16
    d: int = 64
    R: int = 48
    h: float = 0.35
    sigma: float = 0.30
    h_v: float = 0.45
    gamma: float = 0.5
    alpha: float = 1.0
    eta: float = 0.1
    n_attractors: int = 8
    ema_beta: float = 0.2
    n_time_feats: int = 8
    encoder_hidden: int = 256
    potential_hidden: int = 256
    satiation_hidden: int = 128
    satiation_enabled: bool = True
    atom_eps: float = 1.0e-3


@dataclass
class SolverCfg:
    kind: str = "rk4"
    delta_safe: float = 0.5
    r_max: float = 2.0


@dataclass
class RegCfg:
    ess_min_frac: float = 0.25
    lambda_ess: float = 1.0
    lambda_dis: float = 0.1


@dataclass
class TrainCfg:
    epochs: int = 20
    batch_size: int = 192
    bptt: int = 64
    n_negatives: int = 256
    negatives: str = "uniform"
    lr: float = 3.0e-4
    weight_decay: float = 1.0e-2
    warmup_steps: int = 2000
    clip: float = 1.0
    max_len: int = 0              # 0 = no truncation of the train history (v1.2.1)
    ckpt_every_batches: int = 50


@dataclass
class EvalCfg:
    ks: list = field(default_factory=lambda: [10, 50])
    primary_metric: str = "ndcg@10"
    chunk_size: int = 2048
    batch_size: int = 128


@dataclass
class BaselineCfg:
    kind: str = "none"               # none | sasrec | sasrec_psi
    n_layers: int = 2
    n_heads: int = 2
    dropout: float = 0.2
    max_len: int = 200
    eval_windows_per_batch: int = 512


@dataclass
class RunConfig:
    data: DataCfg = field(default_factory=DataCfg)
    semantics: SemanticsCfg = field(default_factory=SemanticsCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    solver: SolverCfg = field(default_factory=SolverCfg)
    reg: RegCfg = field(default_factory=RegCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    baseline: BaselineCfg = field(default_factory=BaselineCfg)
    seed: int = 0
    profile: str = "unset"
    claim_eligible: bool = False
    prereg_version: str = ""
    prereg_sha256: str = ""

    def assert_claim_eligible(self) -> None:
        if not self.claim_eligible:
            raise ConfigError(
                f"run built under profile '{self.profile}' is not claim-eligible; "
                "its results may not enter any paper table")

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))


# ------------------------------------------------------------- key plumbing

def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _resolve(obj: Any, dotted: str):
    parts = dotted.split(".")
    for p in parts[:-1]:
        if not is_dataclass(obj) or p not in {f.name for f in fields(obj)}:
            raise ConfigError(f"unknown config key: {dotted}")
        obj = getattr(obj, p)
    leaf = parts[-1]
    if not is_dataclass(obj) or leaf not in {f.name for f in fields(obj)}:
        raise ConfigError(f"unknown config key: {dotted}")
    cur = getattr(obj, leaf)
    if is_dataclass(cur):
        raise ConfigError(f"{dotted} is a section, not a value")
    return obj, leaf, cur


def _coerce(value, current, key):
    if isinstance(current, bool):
        if not isinstance(value, bool):
            raise ConfigError(f"{key} expects a bool, got {value!r}")
        return value
    if isinstance(current, int):
        if isinstance(value, bool):
            raise ConfigError(f"{key} expects an int, got {value!r}")
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if not isinstance(value, int):
            raise ConfigError(f"{key} expects an int, got {value!r}")
        return value
    if isinstance(current, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{key} expects a number, got {value!r}")
        return float(value)
    if isinstance(current, str):
        if not isinstance(value, str):
            raise ConfigError(f"{key} expects a string, got {value!r}")
        return value
    if isinstance(current, list):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{key} expects a list, got {value!r}")
        return list(value)
    return value


def set_key(cfg: RunConfig, dotted: str, value) -> None:
    obj, leaf, cur = _resolve(cfg, dotted)
    setattr(obj, leaf, _coerce(value, cur, dotted))


def get_key(cfg: RunConfig, dotted: str):
    return _resolve(cfg, dotted)[2]


# ---------------------------------------------------------- preregistration

@dataclass
class Preregistration:
    path: Path
    raw: dict
    sha256: str
    version: str
    frozen: dict
    sweep_keys: list
    grids: dict
    tuning: dict
    acceptance: dict
    baselines: dict
    engineering: dict
    gates: dict
    implementation: dict

    @property
    def implementation_sha256(self) -> str | None:
        return (self.implementation or {}).get("sha256")

    def seeds_confirmatory(self, M: int) -> list:
        """Confirmatory seeds for arm M.  The primary contrast (treat_M, ctrl_M)
        gets more seeds, because the acceptance CI includes between-seed variance."""
        primary = {int(self.acceptance["treat_M"]), int(self.acceptance["ctrl_M"])}
        if int(M) in primary:
            return list(self.grids["seeds_confirmatory_primary"])
        return list(self.grids["seeds_confirmatory"])

    def seeds_all(self) -> list:
        return sorted(set(self.grids["seeds_tuning"]) | set(self.grids["seeds_confirmatory"])
                      | set(self.grids["seeds_confirmatory_primary"]))

    @property
    def sha12(self) -> str:
        return self.sha256[:12]


_REQUIRED = ("version", "frozen", "sweep_keys", "grids", "tuning", "acceptance")


def load_prereg(path) -> Preregistration:
    path = Path(path)
    blob = path.read_bytes()
    raw = yaml.safe_load(blob) or {}
    missing = [k for k in _REQUIRED if k not in raw]
    if missing:
        raise ConfigError(f"preregistration {path} is missing sections {missing}")
    frozen = flatten(raw["frozen"])
    sweep_keys = list(raw["sweep_keys"])
    probe = RunConfig()
    for k, v in frozen.items():
        set_key(probe, k, v)                  # every frozen key must exist
    for k in sweep_keys:
        _resolve(probe, k)                    # every sweep key must exist
    both = sorted(set(frozen) & set(sweep_keys))
    if both:
        raise ConfigError(f"keys cannot be both frozen and swept: {both}")
    if "seeds_confirmatory_primary" not in raw["grids"]:
        raise ConfigError("grids.seeds_confirmatory_primary is required (seed-level inference, v1.2)")
    if set(raw["grids"]["seeds_tuning"]) & (set(raw["grids"]["seeds_confirmatory"])
                                          | set(raw["grids"]["seeds_confirmatory_primary"])):
        raise ConfigError("a tuning seed may not also be a confirmatory seed")
    _verify_amendment_log(path, raw)
    return Preregistration(
        path=path, raw=raw, sha256=sha256_bytes(blob), version=str(raw["version"]),
        frozen=frozen, sweep_keys=sweep_keys, grids=raw["grids"], tuning=raw["tuning"],
        acceptance=raw["acceptance"], baselines=raw.get("baselines", {}),
        engineering=raw.get("engineering", {}), gates=raw.get("gates", {}) or {},
        implementation=raw.get("implementation", {}) or {},
    )


def _verify_amendment_log(path: Path, raw: dict) -> None:
    """The amendment log is part of the contract: its SHA-256 is written in the
    preregistration, so editing the log without re-freezing is detected."""
    doc = raw.get("amendment_log")
    if not doc:
        return
    f = (path.parent / doc["path"]).resolve()
    if not f.exists():
        raise ConfigError(f"amendment log {f} referenced by {path.name} is missing")
    got = sha256_bytes(f.read_bytes())
    if got != doc["sha256"]:
        raise ConfigError(f"amendment log {f.name} changed after freezing "
                          f"(sha256 {got[:12]} != {str(doc['sha256'])[:12]}); re-freeze with a new version")


# ------------------------------------------------------------------ profiles

@dataclass
class Profile:
    name: str
    claim_eligible: bool
    may_override: list
    overrides: dict
    path: Path | None = None


def _allowed(key: str, patterns) -> bool:
    for p in patterns:
        if p.endswith(".*") and key.startswith(p[:-1]):
            return True
        if key == p:
            return True
    return False


def load_profile(path) -> Profile:
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    name = raw.get("name") or path.stem
    claim = bool(raw.get("claim_eligible", False))
    may = list(raw.get("may_override", []) or [])
    over = flatten(raw.get("overrides", {}) or {})
    if claim and (may or over):
        raise ConfigError(
            f"profile '{name}' is claim-eligible but overrides preregistered values; "
            "a claim-eligible profile may not override anything")
    for k in over:
        if not _allowed(k, may):
            raise ConfigError(f"profile '{name}' overrides '{k}' which is not in its may_override list")
    return Profile(name=name, claim_eligible=claim, may_override=may, overrides=over, path=path)


# ------------------------------------------------------------------- builder

def _in_grid(val, grid, name):
    if not any(abs(float(val) - float(g)) < 1e-9 for g in grid):
        raise ConfigError(f"claim-eligible run has {name}={val}, which is outside the preregistered grid {grid}")


def build_run_config(prereg: Preregistration, profile: Profile, sweep: dict | None = None,
                     *, for_training: bool = True) -> RunConfig:
    """for_training=False is used by the data / covariate stages, which need the
    frozen data + semantics values but are not an arm of the grid."""
    cfg = RunConfig()
    for k, v in prereg.frozen.items():
        set_key(cfg, k, v)
    for k, v in profile.overrides.items():
        set_key(cfg, k, v)
    for k, v in (sweep or {}).items():
        if k not in prereg.sweep_keys:
            raise ConfigError(
                f"'{k}' is not a sweep key -- it is frozen by preregistration v{prereg.version}. "
                "Changing it requires a written amendment, not a command-line flag.")
        set_key(cfg, k, v)

    if profile.claim_eligible and for_training:
        g = prereg.grids
        if cfg.baseline.kind == "none":
            _in_grid(cfg.model.M, g["M"], "model.M")
            _in_grid(cfg.model.h, g["h"], "model.h")
            _in_grid(cfg.model.sigma, g["sigma"], "model.sigma")
        else:
            if cfg.baseline.kind not in ("sasrec", "sasrec_psi"):
                raise ConfigError(f"unknown baseline '{cfg.baseline.kind}'")
            _in_grid(cfg.baseline.dropout, prereg.baselines["sasrec_dropout_grid"], "baseline.dropout")
            if cfg.baseline.kind == "sasrec_psi":
                _in_grid(cfg.model.sigma, g["sigma"], "model.sigma")
        if cfg.seed not in prereg.seeds_all():
            raise ConfigError(f"seed {cfg.seed} is not a preregistered seed {prereg.seeds_all()}")
        if (cfg.baseline.kind == "none" and cfg.seed not in g["seeds_tuning"]
                and cfg.seed not in prereg.seeds_confirmatory(cfg.model.M)):
            raise ConfigError(f"seed {cfg.seed} is not preregistered for arm M={cfg.model.M}")
        if cfg.train.negatives not in g.get("negatives", ["uniform"]):
            raise ConfigError(f"negatives '{cfg.train.negatives}' not preregistered")

    cfg.profile = profile.name
    cfg.claim_eligible = profile.claim_eligible
    cfg.prereg_version = prereg.version
    cfg.prereg_sha256 = prereg.sha256
    return cfg

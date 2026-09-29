"""Frozen dataset artifacts produced by `prepare` and consumed by every run."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..common import read_json, sha256_file
from ..config import ARTIFACT_KEYS, ConfigError


@dataclass
class Artifacts:
    root: Path
    manifest: dict
    X: np.ndarray            # [N, d_x] raw content features
    events: pd.DataFrame     # u, i, t, split, session
    popularity: np.ndarray   # train-only counts
    cold: np.ndarray         # bool: first interaction at/after T_cut
    warm: np.ndarray         # bool: seen in train (negative-sampling support)
    n_users: int
    n_items: int


def load_artifacts(data_dir) -> Artifacts:
    root = Path(data_dir)
    if not (root / "manifest.json").exists():
        raise FileNotFoundError(f"{root}/manifest.json not found -- run scripts/prepare_ml25m.py first")
    man = read_json(root / "manifest.json")
    return Artifacts(
        root=root, manifest=man,
        X=np.load(root / "item_features.npy"),
        events=pd.read_parquet(root / "events.parquet"),
        popularity=np.load(root / "popularity.npy"),
        cold=np.load(root / "cold_items.npy"),
        warm=np.load(root / "warm_items.npy"),
        n_users=int(man["n_users"]), n_items=int(man["n_items"]),
    )


def check_artifacts_match(art: Artifacts, cfg, *, strict: bool | None = None,
                          implementation_sha256: str | None = None) -> dict:
    """The artifact must have been built by THIS code schema under THIS
    preregistration, with the same dataset-defining keys.  Equal data_cfg is not
    enough: v1.1 and v1.2 share data_cfg but define cold items differently.
    For a claim-eligible run any disagreement is fatal."""
    from .prepare import DATA_SCHEMA
    stored = art.manifest["data_cfg"]
    mine = asdict(cfg.data)
    diff = {k: (stored.get(k), mine[k]) for k in ARTIFACT_KEYS if stored.get(k) != mine[k]}
    if art.manifest.get("schema") != DATA_SCHEMA:
        diff["schema"] = (art.manifest.get("schema"), DATA_SCHEMA)
    if art.manifest.get("prereg_sha256") != cfg.prereg_sha256:
        diff["prereg_sha256"] = (str(art.manifest.get("prereg_sha256"))[:12], cfg.prereg_sha256[:12])
    if int(art.manifest.get("n_users", -1)) != int(cfg.data.n_users):
        diff["actual_n_users"] = (art.manifest.get("n_users"), cfg.data.n_users)
    if implementation_sha256 and art.manifest.get("implementation_sha256") != implementation_sha256:
        diff["implementation_sha256"] = (str(art.manifest.get("implementation_sha256"))[:12],
                                         implementation_sha256[:12])
    strict = cfg.claim_eligible if strict is None else strict
    if diff and strict:
        raise ConfigError(f"artifact at {art.root} does not belong to this preregistration/code "
                          f"({diff}); rebuild it with scripts/prepare_ml25m.py")
    return diff


def _verify_files(root: Path, files: dict, what: str) -> None:
    if not files:
        raise ConfigError(f"{what} manifest in {root} lists no payload hashes (built by older code)")
    for name, sha in files.items():
        f = root / name
        if not f.exists() or sha256_file(f) != sha:
            raise ConfigError(f"{what} payload {f} is missing or was modified after it was built")


def verify_data_payload(data_dir, prereg, *, strict: bool = True) -> dict:
    """Every payload file matches the hash recorded at build time; in strict mode the
    artifact also belongs to this preregistration, schema and implementation."""
    from ..provenance import expected_implementation
    from .prepare import DATA_SCHEMA
    root = Path(data_dir)
    man = read_json(root / "manifest.json")
    _verify_files(root, man.get("files"), "data")
    if strict:
        bad = {k: v for k, v in (("schema", (man.get("schema"), DATA_SCHEMA)),
                                 ("prereg_sha256", (man.get("prereg_sha256"), prereg.sha256)),
                                 ("implementation_sha256", (man.get("implementation_sha256"),
                                                            expected_implementation(prereg))))
               if v[0] != v[1]}
        if bad:
            raise ConfigError(f"data artifact {root} does not belong to this protocol: {bad}")
    return man


def verify_semantics(data_dir, prereg, semantics_cfg: dict) -> dict:
    """The H1 moderator itself: covariates.parquet (lambda_hat, tau^2, strata, tail),
    clusters and head items must be byte-identical to what was built -- by this
    implementation, under this preregistration, from THIS data artifact."""
    from ..provenance import expected_implementation
    from .covariates import SEMANTICS_SCHEMA
    root = Path(data_dir)
    sem = read_json(root / "semantics_manifest.json")
    _verify_files(root, sem.get("files"), "semantics")
    checks = {
        "schema": (sem.get("schema"), SEMANTICS_SCHEMA),
        "prereg_sha256": (sem.get("prereg_sha256"), prereg.sha256),
        "implementation_sha256": (sem.get("implementation_sha256"), expected_implementation(prereg)),
        "data_manifest_sha256": (sem.get("data_manifest_sha256"), sha256_file(root / "manifest.json")),
        "semantics_cfg": (sem.get("semantics_cfg"), semantics_cfg),
    }
    bad = {k: v for k, v in checks.items() if v[0] != v[1]}
    if bad:
        raise ConfigError(f"covariates in {root} do not match this protocol: {sorted(bad)}")
    return sem

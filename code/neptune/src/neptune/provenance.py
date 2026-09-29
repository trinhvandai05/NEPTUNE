"""Implementation fingerprint: the preregistration locks the DESIGN; this locks the CODE.

sha256 over (relative path, file sha256) of every file that can change a result:
src/neptune/**/*.py, scripts/*.py and pyproject.toml.  Configs and AMENDMENTS.md
are excluded -- they are locked by the preregistration hash itself.

The frozen value lives in the preregistration (`implementation.sha256`).  Claim
runs, the OPEN-1 pilot and the analysis refuse to proceed on different code, and
every run records the fingerprint it ran with, so runs made by two versions of
the code can never be mixed or silently reused.  Git state is recorded too, but
not required (the package may be used from a zip).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .common import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]


def implementation_fingerprint(root: Path | None = None) -> dict:
    root = Path(root or REPO_ROOT)
    files = list((root / "src" / "neptune").rglob("*.py")) + list((root / "scripts").glob("*.py"))
    files += [root / "pyproject.toml"]
    files = sorted(f for f in files if f.exists() and "__pycache__" not in f.parts)
    h, per = hashlib.sha256(), {}
    for f in files:
        rel = f.relative_to(root).as_posix()
        d = sha256_file(f)
        per[rel] = d
        h.update(rel.encode() + b"\0" + d.encode() + b"\n")
    return {"sha256": h.hexdigest(), "n_files": len(files), "files": per}


def implementation_sha256() -> str:
    return implementation_fingerprint()["sha256"]


def expected_implementation(prereg) -> str:
    """The fingerprint every protocol artifact must carry: the frozen one if the
    preregistration freezes one, else the code running right now."""
    return prereg.implementation_sha256 or implementation_sha256()


def check_implementation(prereg, strict: bool = True) -> str:
    """Current fingerprint; in strict mode it must equal the frozen one.  Called by
    training (claim + pilot), the OPEN-1 decision, selection and the analysis -- the
    code that COMPUTES the verdict is locked, not only the code that trains."""
    from .config import ConfigError
    current = implementation_sha256()
    frozen = prereg.implementation_sha256
    if strict and frozen and current != frozen:
        raise ConfigError(f"code changed after freezing: implementation {current[:12]} != preregistered "
                          f"{frozen[:12]}. Revert, or re-freeze under a new preregistration version.")
    return current

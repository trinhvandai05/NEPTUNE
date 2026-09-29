"""Run provenance: git, environment, hardware, seed, preregistration hash."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

from ..common import write_json_atomic


def git_info(start: Path | None = None) -> dict:
    cwd = str(start or Path(__file__).resolve().parent)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True,
                                text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=cwd, capture_output=True,
                               text=True, timeout=10).stdout.strip()
        return {"available": bool(commit), "commit": commit or None, "dirty": bool(dirty)}
    except Exception as e:
        return {"available": False, "error": str(e)}


def environment_info() -> dict:
    import numpy
    import pandas
    import sklearn
    import torch
    return {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "numpy": numpy.__version__, "pandas": pandas.__version__, "sklearn": sklearn.__version__,
    }


def hardware_info(device) -> dict:
    import torch
    info = {"device": str(device), "cpu_count": os.cpu_count()}
    if str(device).startswith("cuda") and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update({"gpu": torch.cuda.get_device_name(0),
                     "gpu_memory_gb": round(props.total_memory / 1e9, 2)})
    else:
        info["gpu"] = None
    return info


def write_run_manifests(run_dir: Path, cfg, device) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.yaml")
    (run_dir / "preregistration_sha256.txt").write_text(cfg.prereg_sha256 + "\n")
    write_json_atomic(run_dir / "git.json", git_info())
    write_json_atomic(run_dir / "environment.json", environment_info())
    write_json_atomic(run_dir / "hardware.json", hardware_info(device))
    write_json_atomic(run_dir / "seed.json", {"seed": cfg.seed})

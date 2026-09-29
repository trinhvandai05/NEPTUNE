"""Shared CLI plumbing for scripts/*.py (keeps each script a thin wrapper)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:          # works without `pip install -e .`
    sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from neptune.config import load_prereg, load_profile  # noqa: E402


def parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--prereg", default=str(ROOT / "configs" / "preregistered_v1_2_3.yaml"))
    p.add_argument("--profile", default=str(ROOT / "configs" / "profiles" / "h1_claim.yaml"))
    p.add_argument("--data", default=None,
                   help="dataset artifact dir (default: artifacts/data/<profile>/<prereg sha12>)")
    p.add_argument("--out", default=str(ROOT / "artifacts"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    return p


def setup(args):
    if args.threads:
        torch.set_num_threads(args.threads)
    if str(args.device).startswith("cuda"):
        # Keep TF32 OFF: kernel exponents (cos - 1)/h^2 amplify TF32's ~1e-3 cosine
        # error by 1/h^2 (25x at h = 0.2).  The kernel math must stay true fp32.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    prereg, profile = load_prereg(args.prereg), load_profile(args.profile)
    if args.data is None:     # namespaced like runs: artifacts of another prereg are never picked up
        args.data = str(Path(args.out) / "data" / profile.name / prereg.sha12)
    print(f"[neptune] prereg v{prereg.version} ({prereg.sha12}) · profile {profile.name} · data {args.data}")
    return prereg, profile


def selection_dir(args, profile, prereg) -> Path:
    from neptune.logging.registry import selection_dir as _sd
    return _sd(args.out, profile.name, prereg.sha256)


def run_all(prereg, profile, sweeps, args, experiment):
    from neptune.pipeline.runner import execute
    n = len(sweeps)
    for k, s in enumerate(sweeps, 1):
        print(f"\n=== [{experiment}] run {k}/{n}: {s}")
        execute(prereg, profile, s, data_dir=args.data, out_root=args.out,
                device=args.device, experiment=experiment)

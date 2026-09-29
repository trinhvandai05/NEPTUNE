#!/usr/bin/env python3

import argparse
from pathlib import Path

from neptune.config import load_prereg, load_profile
from neptune.pipeline.runner import execute
from neptune.logging.registry import EXPERIMENTS


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code" / "neptune"


parser = argparse.ArgumentParser()

parser.add_argument("--data", required=True)
parser.add_argument("--out", default=str(ROOT / "artifacts"))

parser.add_argument(
    "--profile",
    default=str(ROOT / "profiles" / "h1_preview_2027.yaml")
)

parser.add_argument("--M", type=int, required=True)
parser.add_argument("--h", type=float, required=True)
parser.add_argument("--sigma", type=float, default=0.30)
parser.add_argument("--seed", type=int, default=2027)

parser.add_argument(
    "--negatives",
    choices=["uniform", "popularity"],
    default="uniform"
)

parser.add_argument(
    "--time",
    choices=["ordinal", "logdt"],
    default="ordinal"
)

parser.add_argument("--device", default="cuda")

args = parser.parse_args()


prereg = load_prereg(
    CODE / "configs" / "preregistered_v1_2_3.yaml"
)

profile = load_profile(args.profile)


EXPERIMENT = "H1_LOCAL_PREVIEW"

EXPERIMENTS[EXPERIMENT] = (
    "claim-ineligible exploratory local-server H1 run"
)


sweep = {
    "model.M": args.M,
    "model.h": args.h,
    "model.sigma": args.sigma,
    "seed": args.seed,
    "train.negatives": args.negatives,
    "data.ordinal_time": args.time == "ordinal",
}


print("=" * 80)
print(
    f"TRAIN M={args.M} "
    f"h={args.h} "
    f"sigma={args.sigma} "
    f"seed={args.seed}"
)
print("DATA:", args.data)
print("OUT :", args.out)
print("=" * 80)


summary = execute(
    prereg,
    profile,
    sweep,
    data_dir=args.data,
    out_root=args.out,
    device=args.device,
    experiment=EXPERIMENT,
    resume=True,
    log=print,
)


print("\nFINISHED")
print("M       =", args.M)
print("h       =", args.h)
print("sigma   =", args.sigma)
print("seed    =", args.seed)
print(
    summary["val"]["metric"],
    "=",
    summary["val"]["value"]
)
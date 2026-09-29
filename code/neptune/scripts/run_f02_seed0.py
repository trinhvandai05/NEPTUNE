#!/usr/bin/env python
"""F02_SEED0_TUNING -- every M arm at seed 0 over its candidate grid (reuses F01 runs)."""
from _cli import parser, run_all, selection_dir, setup

from neptune.common import read_json
from neptune.pipeline.plans import f02_seed0_sweeps

args = parser(__doc__).parse_args()
prereg, profile = setup(args)
policy = read_json(selection_dir(args, profile, prereg) / "sigma_policy.json")
run_all(prereg, profile, f02_seed0_sweeps(prereg, policy), args, "F02_SEED0_TUNING")

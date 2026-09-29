#!/usr/bin/env python
"""F01_BANDWIDTH_PROBE -- M in f01_M x h x sigma at seed 0 (validation only)."""
from _cli import parser, run_all, setup

from neptune.pipeline.plans import f01_sweeps

args = parser(__doc__).parse_args()
prereg, profile = setup(args)
run_all(prereg, profile, f01_sweeps(prereg), args, "F01_BANDWIDTH_PROBE")

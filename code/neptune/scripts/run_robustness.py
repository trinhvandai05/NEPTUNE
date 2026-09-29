#!/usr/bin/env python
"""Robustness arms at the frozen h*(M): log-compressed time (NOT raw Delta_t), optional popularity negatives."""
from _cli import parser, run_all, selection_dir, setup

from neptune.pipeline.selection import verify_frozen_selection
from neptune.pipeline.plans import robustness_sweeps, secondary_negatives_sweeps

p = parser(__doc__)
p.add_argument("--with-popularity-negatives", action="store_true")
args = p.parse_args()
prereg, profile = setup(args)
sel = verify_frozen_selection(prereg, args.out, profile.name, args.data)   # raises if anything drifted
run_all(prereg, profile, robustness_sweeps(prereg, sel), args, "H1_ROBUSTNESS_LOGDT")
if args.with_popularity_negatives:
    run_all(prereg, profile, secondary_negatives_sweeps(prereg, sel), args, "H1_ROBUSTNESS_POPNEG")

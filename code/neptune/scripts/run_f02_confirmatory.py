#!/usr/bin/env python
"""F02_CONFIRMATORY -- confirmatory seeds at the frozen h*(M). No re-selection."""
from _cli import parser, run_all, selection_dir, setup

from neptune.pipeline.selection import verify_frozen_selection
from neptune.pipeline.plans import confirmatory_sweeps

args = parser(__doc__).parse_args()
prereg, profile = setup(args)
sel = verify_frozen_selection(prereg, args.out, profile.name, args.data)   # raises if anything drifted
run_all(prereg, profile, confirmatory_sweeps(prereg, sel), args, "F02_CONFIRMATORY")

#!/usr/bin/env python
"""P00_PILOT -- claim-INELIGIBLE pilot on ML-25M for the OPEN-1 gate:
M in f01_M x h in {min, max} at the middle sigma, tuning seed, CLAIM epoch count.
Then run scripts/pilot_report.py, which freezes the OPEN-1 decision."""
from _cli import ROOT, parser, run_all, setup

from neptune.pipeline.gates import pilot_sweeps

p = parser(__doc__)
p.set_defaults(profile=str(ROOT / "configs" / "profiles" / "engineering_pilot.yaml"))
args = p.parse_args()
prereg, profile = setup(args)
if profile.claim_eligible:
    raise SystemExit("the pilot must run under a claim-INELIGIBLE profile")
run_all(prereg, profile, pilot_sweeps(prereg), args, "P00_PILOT")

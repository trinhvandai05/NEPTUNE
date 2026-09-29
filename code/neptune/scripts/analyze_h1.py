#!/usr/bin/env python
"""H1_ANALYSIS -- unseal test for the frozen selection only; D1, D2-A..D, conditions, verdict."""
from _cli import parser, setup

from neptune.pipeline.analyze import analyze_h1

args = parser(__doc__).parse_args()
prereg, profile = setup(args)
rep = analyze_h1(prereg, profile, data_dir=args.data, out_root=args.out)
print(f"\nconditions: {rep['acceptance']['conditions']}")
print(f"VERDICT: {rep['acceptance']['verdict']}")
from neptune.logging.registry import report_dir  # noqa: E402
print(f"report: {report_dir(args.out, profile.name, prereg.sha256) / 'h1_report.md'}")

#!/usr/bin/env python
"""Decide shared vs per-arm sigma from F01 validation results (cost rule)."""
from _cli import parser, selection_dir, setup

from neptune.pipeline.selection import collect_val_table, decide_sigma_policy, freeze_policy

args = parser(__doc__).parse_args()
prereg, profile = setup(args)
g, tun = prereg.grids, prereg.tuning
val = collect_val_table(args.out, profile.name, prereg.sha256, metric=tun["selection_metric"],
                        negatives=g["negatives_primary"], seeds=tuple(g["seeds_tuning"]))
pol = decide_sigma_policy(val[val["M"].isin(g["f01_M"])], g["f01_M"], g["sigma"], g["h"],
                          tun["sigma_cost_threshold"])
pol = freeze_policy(pol, val, g["f01_M"], prereg, selection_dir(args, profile, prereg) / "sigma_policy.json")
print(f"sigma policy: {pol['mode']}  shared sigma={pol['shared_sigma']}  cost by arm={pol['cost_by_arm']}")

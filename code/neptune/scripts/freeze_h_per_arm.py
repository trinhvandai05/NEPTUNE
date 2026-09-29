#!/usr/bin/env python
"""Freeze h*(M) (and sigma*(M) under the per-arm policy) from VALIDATION only. Write-once."""
from _cli import parser, selection_dir, setup

from neptune.common import read_json
from neptune.pipeline.selection import anchor_selection, collect_val_table, freeze_selection, select_per_arm

args = parser(__doc__).parse_args()
prereg, profile = setup(args)
g, tun = prereg.grids, prereg.tuning
sdir = selection_dir(args, profile, prereg)
policy = read_json(sdir / "sigma_policy.json")
val = collect_val_table(args.out, profile.name, prereg.sha256, metric=tun["selection_metric"],
                        negatives=g["negatives_primary"], seeds=tuple(g["seeds_tuning"]))
sel = select_per_arm(val, policy, g["M"], g["h"], g["sigma"], tun["selection_metric"])
sel = freeze_selection(anchor_selection(sel, prereg, args.data), sdir / "selected_h.json")
for M, a in sel["arms"].items():
    print(f"M={M:>3}: h*={a['h']:.2f} sigma={a['sigma']:.2f} val={a['val_value']:.4f}")

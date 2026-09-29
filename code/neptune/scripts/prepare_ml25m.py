#!/usr/bin/env python
"""I00_DATA -- download (or reuse) ML-25M and build the frozen artifacts."""
from _cli import parser, setup

from neptune.config import build_run_config
from neptune.data.download import download_ml25m
from neptune.pipeline.prepare import prepare_dataset

p = parser(__doc__)
p.add_argument("--raw-dir", default=None, help="already-extracted ml-25m directory (skips download)")
p.add_argument("--download-root", default="artifacts/raw")
p.add_argument("--no-verify", action="store_true", help="skip the md5 check (not recommended)")
args = p.parse_args()
prereg, profile = setup(args)
cfg = build_run_config(prereg, profile, for_training=False)
raw = args.raw_dir or download_ml25m(args.download_root, verify=not args.no_verify)
man = prepare_dataset(raw, args.data, cfg)
print(f"users={man['n_users']} items={man['n_items']} events={man['n_events']} "
      f"split={man['split_counts']}")
print("attrition:", man["attrition"])
print(f"cold items in test: {man['n_cold_items_in_test']} (cold arm usable: {man['cold_arm_usable']})")

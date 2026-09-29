#!/usr/bin/env python
"""BENCH -- staged throughput, oracle, E_vec, ratchet (enforced on the reference GPU only)."""
import json
import sys
from pathlib import Path

from _cli import parser, setup

from neptune.benchmarks.harness import ratchet, run_benchmarks
from neptune.common import write_json_atomic
from neptune.config import build_run_config
from neptune.logging.manifest import hardware_info

p = parser(__doc__)
p.add_argument("--M", type=int, default=16)
p.add_argument("--stages", default="kernel,state,rk4,train")
p.add_argument("--enforce", action="store_true", help="exit 1 if the ratchet fails on the reference GPU")
args = p.parse_args()
prereg, profile = setup(args)
cfg = build_run_config(prereg, profile, {"model.M": args.M}, for_training=False)
res = run_benchmarks(cfg, prereg.engineering["benchmark"], args.device, tuple(args.stages.split(",")))
hw = hardware_info(args.device)
rat = ratchet(res, prereg.engineering["ratchet_events_per_sec"], hw.get("gpu"),
              prereg.engineering["reference_hardware"])
out = {"results": res, "ratchet": rat, "hardware": hw, "prereg_sha256": prereg.sha256}
write_json_atomic(Path(args.out) / "benchmarks" / f"bench_M{args.M}.json", out)
print(json.dumps({k: (v["events_per_sec"] if isinstance(v, dict) and "events_per_sec" in v else v)
                  for k, v in res.items()}, indent=2))
print("ratchet:", json.dumps(rat["checks"], indent=2), "| enforced:", rat["enforced"])
if args.enforce and rat["enforced"] and not rat["all_pass"]:
    sys.exit(1)

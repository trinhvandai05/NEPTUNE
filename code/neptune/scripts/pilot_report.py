#!/usr/bin/env python
"""Summarize the OPEN-1 pilot of THIS preregistration only and FREEZE the decision
(artifacts/gates/<sha12>/open1_decision.json, write-once).  Every claim-eligible
run refuses to start unless that file says KEEP_CURRENT_PREREG."""
import json
from pathlib import Path

from _cli import ROOT, parser, setup

import numpy as np
import pandas as pd

from neptune.common import read_json
from neptune.config import build_run_config
from neptune.logging.registry import run_dir
from neptune.pipeline.gates import decide_open1, freeze_open1, gate_path, pilot_sweeps

p = parser(__doc__)
p.set_defaults(profile=str(ROOT / "configs" / "profiles" / "engineering_pilot.yaml"))
p.add_argument("--dry-run", action="store_true", help="print the decision without freezing it")
args = p.parse_args()
prereg, profile = setup(args)

rows = []
for sw in pilot_sweeps(prereg):                       # this namespace only
    rdir = run_dir(args.out, build_run_config(prereg, profile, sw))
    if not (rdir / "summary.json").exists():
        continue
    s = read_json(rdir / "summary.json")
    m = pd.read_parquet(rdir / "metrics.parquet")
    mean_of = lambda k: float(m.loc[m["metric"] == k, "value"].mean())
    rows.append({"run": rdir.name, "val": s["val"]["value"], "ev_per_s": s.get("train_events_per_sec_median"),
                 "peak_vram_gb": s.get("peak_vram_gb"), "D17_final": s["D17_final"]["mean_pairwise_cos"],
                 "eff_rank": s["D17_final"]["effective_rank"], "ess_mean": mean_of("ess_mean"),
                 "atom_evicted_rate": mean_of("atom_evicted_rate"), "dissipation": mean_of("dissipation")})
pd.set_option("display.width", 200)
print(pd.DataFrame(rows).round(4).to_string(index=False) if rows else "no finished pilot runs in this namespace")

cov_p = Path(args.data) / "covariates.parquet"
if cov_p.exists():                                    # compute impact of full-history training
    cov = pd.read_parquet(cov_p)
    long = cov["n_train"] > 1024
    print(f"\nhistory length: median {cov['n_train'].median():.0f}, p99 {cov['n_train'].quantile(.99):.0f}, "
          f"max {cov['n_train'].max()}; users > 1024: {long.mean():.2%}, "
          f"their share of train events: {cov.loc[long, 'n_train'].sum() / cov['n_train'].sum():.2%}")
    print(cov.groupby("length_stratum")["n_train"].agg(["min", "median", "max"]).to_string())
man = Path(args.data) / "manifest.json"
if man.exists():
    print("\nattrition:", json.dumps(read_json(man)["attrition"], indent=1))

decision = decide_open1(prereg, profile, args.out)
print("\nD17 trajectories:")
for r in decision["runs"]:
    print(f"  M={r['M']:>3} h={r['h']:.2f}: " + " ".join(f"{v:.3f}" for v in r["D17_trajectory"]))
print(f"differential D17 across M (per h): {decision['differential_D17_across_M']}")
print(f"max final D17 = {decision['max_D17_final']:.3f}  (threshold {decision['threshold']})")
print(f"OPEN-1 VERDICT: {decision['verdict']}")
if not args.dry_run:
    freeze_open1(decision, gate_path(args.out, prereg))
    print(f"frozen: {gate_path(args.out, prereg)}")

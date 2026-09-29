#!/usr/bin/env python
"""Run the WHOLE H1 pipeline on synthetic ML-25M-format data with the tiny smoke
preregistration.  ~1-3 min on CPU.  Verifies an installation end to end:

  synthetic raw -> prepare -> covariates -> F01 -> sigma policy -> F02 seed 0
  -> freeze h*(M) -> confirmatory -> robustness -> analysis -> baselines -> bench

The verdict it prints is about synthetic data and means nothing scientifically.
"""
import argparse
import shutil
from pathlib import Path

import _cli  # noqa: F401
import torch

from neptune.benchmarks.harness import run_benchmarks
from neptune.common import read_json, write_json_atomic
from neptune.config import build_run_config, load_prereg, load_profile
from neptune.pipeline.analyze import analyze_h1
from neptune.pipeline.gates import decide_open1, freeze_open1, gate_path, pilot_sweeps
from neptune.pipeline.covariates import build_covariates
from neptune.logging.registry import report_dir, selection_dir
from neptune.pipeline.plans import (baseline_tuning_sweeps, confirmatory_sweeps, f01_sweeps,
                                    f02_seed0_sweeps, robustness_sweeps, secondary_negatives_sweeps)
from neptune.pipeline.prepare import prepare_dataset
from neptune.pipeline.runner import execute
from neptune.pipeline.selection import (anchor_selection, collect_val_table, decide_sigma_policy,
                                        freeze_policy, freeze_selection, select_per_arm,
                                        verify_frozen_selection)
from neptune.synthetic import write_synthetic_ml25m

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--work", default=str(ROOT / "artifacts" / "smoke"))
p.add_argument("--threads", type=int, default=4)
p.add_argument("--skip-baselines", action="store_true")
p.add_argument("--device", default="cpu", help="run the smoke on cuda to check GPU code paths")
a = p.parse_args()
torch.set_num_threads(a.threads)
work = Path(a.work)
shutil.rmtree(work, ignore_errors=True)
prereg = load_prereg(ROOT / "tests" / "fixtures" / "prereg_smoke.yaml")
prof = load_profile(ROOT / "tests" / "fixtures" / "smoke_claim.yaml")
data, out, dev = work / "data", work / "out", a.device
kw = dict(data_dir=data, out_root=out, device=dev, log=lambda *x: None)
g, tun = prereg.grids, prereg.tuning


def step(msg):
    print(f"\n>>> {msg}")


step("synthetic raw data + prepare + covariates")
raw = write_synthetic_ml25m(work / "raw", n_users=300)
base = build_run_config(prereg, prof, for_training=False)
man = prepare_dataset(raw, data, base)
_, sm = build_covariates(data, base)
print(f"users={man['n_users']} items={man['n_items']} events={man['n_events']} deff_median={sm['deff_median']:.2f}")

step("OPEN-1 pilot (claim-ineligible) + frozen gate decision")
pilot = load_profile(ROOT / "tests" / "fixtures" / "smoke_pilot.yaml")
for s in pilot_sweeps(prereg):
    execute(prereg, pilot, s, experiment="P00_PILOT", **kw)
dec = freeze_open1(decide_open1(prereg, pilot, out), gate_path(out, prereg))
print(f"OPEN-1: max D17 {dec['max_D17_final']:.3f} -> {dec['verdict']}")

step(f"F01: {len(f01_sweeps(prereg))} runs")
for s in f01_sweeps(prereg):
    execute(prereg, prof, s, experiment="F01_BANDWIDTH_PROBE", **kw)
sdir = selection_dir(out, prof.name, prereg.sha256)
val = collect_val_table(out, prof.name, prereg.sha256, metric=tun["selection_metric"], seeds=tuple(g["seeds_tuning"]))
pol = decide_sigma_policy(val[val["M"].isin(g["f01_M"])], g["f01_M"], g["sigma"], g["h"], tun["sigma_cost_threshold"])
pol = freeze_policy(pol, val, g["f01_M"], prereg, sdir / "sigma_policy.json")
print(f"sigma policy: {pol['mode']} (costs {pol['cost_by_arm']})")

step(f"F02 seed 0: {len(f02_seed0_sweeps(prereg, pol))} runs (F01 duplicates reused)")
for s in f02_seed0_sweeps(prereg, pol):
    execute(prereg, prof, s, experiment="F02_SEED0_TUNING", **kw)
val = collect_val_table(out, prof.name, prereg.sha256, metric=tun["selection_metric"], seeds=tuple(g["seeds_tuning"]))
sel = freeze_selection(anchor_selection(select_per_arm(val, pol, g["M"], g["h"], g["sigma"],
                                                      tun["selection_metric"]), prereg, data),
                       sdir / "selected_h.json")
sel = verify_frozen_selection(prereg, out, prof.name, data)
print({M: (v["h"], v["sigma"]) for M, v in sel["arms"].items()})

step("confirmatory (tuning seed excluded) + robustness")
for s in confirmatory_sweeps(prereg, sel):
    execute(prereg, prof, s, experiment="F02_CONFIRMATORY", **kw)
for s in robustness_sweeps(prereg, sel):
    execute(prereg, prof, s, experiment="H1_ROBUSTNESS_LOGDT", **kw)
for s in secondary_negatives_sweeps(prereg, sel):
    execute(prereg, prof, s, experiment="H1_ROBUSTNESS_POPNEG", **kw)

step("analysis")
rep = analyze_h1(prereg, prof, data_dir=data, out_root=out)
print(rep["acceptance"]["conditions"], "->", rep["acceptance"]["verdict"])
print("procedural:", rep["acceptance"]["procedural"])

if not a.skip_baselines:
    step("baselines (SASRec, SASRec+Psi)")
    for s in baseline_tuning_sweeps(prereg, "sasrec")[:1] + baseline_tuning_sweeps(prereg, "sasrec_psi", dropout=0.1)[:1]:
        r = execute(prereg, prof, s, experiment="B01_SASREC" if s["baseline.kind"] == "sasrec" else "B02_SASREC_PSI", **kw)
        print(s["baseline.kind"], "val", round(r["val"]["value"], 4))

step("benchmark (tiny, CPU)")
res = run_benchmarks(build_run_config(prereg, prof, {"model.M": 4}, for_training=False),
                     prereg.engineering["benchmark"], dev)
print({k: round(v["events_per_sec"]) for k, v in res.items() if isinstance(v, dict)},
      "E_vec(train)=", round(res["train_E_vec"], 3))
print(f"\nSMOKE OK. Report: {report_dir(out, prof.name, prereg.sha256) / 'h1_report.md'}")

#!/usr/bin/env python
"""B01/B02 -- SASRec (3 dropout trials) then SASRec+Psi (3 sigma trials), val selection, confirmatory seeds."""
from _cli import parser, run_all, selection_dir, setup

from neptune.common import write_json_atomic
from neptune.pipeline.plans import baseline_tuning_sweeps
from neptune.pipeline.runner import execute

args = parser(__doc__).parse_args()
prereg, profile = setup(args)
kw = dict(data_dir=args.data, out_root=args.out, device=args.device)


def pick(sweeps, exp):
    res = [(execute(prereg, profile, s, experiment=exp, **kw)["val"]["value"], s) for s in sweeps]
    return max(res, key=lambda r: r[0])[1]


best_sas = pick(baseline_tuning_sweeps(prereg, "sasrec"), "B01_SASREC")
best_psi = pick(baseline_tuning_sweeps(prereg, "sasrec_psi", dropout=best_sas["baseline.dropout"]), "B02_SASREC_PSI")
write_json_atomic(selection_dir(args, profile, prereg) / "baselines_selected.json",
                  {"selection_split": "val", "sasrec": best_sas, "sasrec_psi": best_psi})
for best, exp in ((best_sas, "B01_SASREC"), (best_psi, "B02_SASREC_PSI")):
    run_all(prereg, profile, [dict(best, seed=s) for s in prereg.grids["seeds_confirmatory"]], args, exp)

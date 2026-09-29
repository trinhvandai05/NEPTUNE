"""Acceptance conditions and verdict -- executed as code, not read off a plot.

C1  capacity        two-level CI lower > 0 AND every confirmatory seed's mean Delta > 0
C2  multimodality   D2-B 'lambda': beta_lambda < 0            (two-level CI upper < 0)
C3  + length        D2-B 'lambda+logn': beta_lambda < 0       (CI upper < 0)
C4  + popularity    D2-B 'lambda+logn+tail_test': beta < 0    (CI upper < 0)
C5  not a length artifact (see below)
C6  procedural      per-arm h chosen on validation, sigma policy applied, every run
                    claim-eligible, tuning seed excluded, D17 manifold check passed

Condition 5 -- direction CORRECTED relative to the draft wording.
The draft said the within-length pattern "must NOT be explained by the
reliability pattern".  That is backwards.  In the world where H1 is true and
uniform across history lengths, measurement error alone makes
beta_j = beta * R_j: the slope looks weak in short strata and strong in long
ones, and beta_j / R_j is constant.  The draft wording would FAIL that world.
The suspicious world is the one where beta_j / R_j itself grows with length
(the effect lives only in long histories beyond what precision explains).
So C5 passes iff, at fixed history length,
  (a) the stratum-fixed-effects slope on lambda_hat is < 0     (CI upper < 0)
  (b) no reliable stratum (R_j >= r_min) has a significantly POSITIVE beta_j
  (c) beta_j / R_j over reliable strata is negative with CV <= cv_max
      (needs >= 2 reliable strata; otherwise C5 cannot pass).
corr(beta_j, R_j) is reported but not gating.

Verdicts
  INVALID_PROTOCOL  C6 fails -- nothing may be claimed
  STOP              not C1 and not C2 (preregistered kill: no capacity gain AND flat curve)
  SUPPORTED         C1..C5
  SUGGESTIVE        C1..C4, not C5 (multimodality not separated from length/precision)
  CAPACITY_ONLY     C1 but not C2 (M > 1 helps, but not where multimodality predicts)
  INCONCLUSIVE      anything else
"""

from __future__ import annotations

import numpy as np


def _hi_below_zero(x) -> bool:
    return bool(x is not None and x.get("ok", True) and np.isfinite(x.get("hi", np.nan)) and x["hi"] < 0)


def evaluate_acceptance(analysis: dict, acceptance: dict, procedural: dict) -> dict:
    """All gating CIs are two-level (user + training seed), see h1_analysis."""
    c = {}
    c1 = analysis["C1"]
    c["C1_capacity"] = bool(np.isfinite(c1["lo"]) and c1["lo"] > 0 and c1["sign_consistent"])
    b = analysis["D2B"]
    c["C2_multimodality"] = _hi_below_zero(b["lambda"])
    c["C3_controls_length"] = _hi_below_zero(b["lambda+logn"])
    c["C4_controls_popularity"] = _hi_below_zero(b["lambda+logn+tail_test"])

    d = analysis["D2C"]
    reliable = [s for s in d["strata"] if s["reliable"]]
    c5a = _hi_below_zero(d["fe_combined"])
    c5b = not any(np.isfinite(s["lo"]) and s["lo"] > 0 for s in reliable)
    c5c = bool(d["n_reliable"] >= 2 and np.isfinite(d["ratio_cv"]) and d["ratio_mean"] < 0
               and d["ratio_cv"] <= acceptance["c5_cv_max"])
    c["C5_not_length_artifact"] = bool(c5a and c5b and c5c)
    c5_detail = {"fe_slope_negative": c5a, "no_reliable_stratum_positive": c5b,
                 "ratio_negative_and_stable": c5c, "n_reliable": d["n_reliable"],
                 "ratio_mean": d["ratio_mean"], "ratio_cv": d["ratio_cv"],
                 "cv_max": acceptance["c5_cv_max"], "r_min": acceptance["c5_r_min"]}

    c["C6_procedural"] = bool(all(procedural.values())) if procedural else False

    if not c["C6_procedural"]:
        verdict = "INVALID_PROTOCOL"
    elif not c["C1_capacity"] and not c["C2_multimodality"]:
        verdict = "STOP"
    elif all(c[k] for k in ("C1_capacity", "C2_multimodality", "C3_controls_length",
                            "C4_controls_popularity", "C5_not_length_artifact")):
        verdict = "SUPPORTED"
    elif all(c[k] for k in ("C1_capacity", "C2_multimodality", "C3_controls_length",
                            "C4_controls_popularity")):
        verdict = "SUGGESTIVE"
    elif c["C1_capacity"] and not c["C2_multimodality"]:
        verdict = "CAPACITY_ONLY"
    else:
        verdict = "INCONCLUSIVE"
    return {"conditions": c, "C5_detail": c5_detail, "procedural": procedural, "verdict": verdict}

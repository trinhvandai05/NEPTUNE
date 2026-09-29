"""Render the H1 report as Markdown (tables first, verdict last)."""

from __future__ import annotations

import numpy as np


def _f(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def _tl(x):
    """two-level estimate: est [lo, hi] (sd_seed, per-seed)"""
    if not x or not x.get("ok", True):
        return "— (not estimable)"
    ps = ", ".join(_f(v, 4) for v in x.get("per_seed", []))
    return (f"{_f(x['est'])} [{_f(x['lo'])}, {_f(x['hi'])}] · sd_seed {_f(x['sd_seed'])} · "
            f"t df={x['df']} · per seed: {ps}")


def render_markdown(an: dict, acc: dict, meta: dict) -> str:
    L = [f"# H1 report — {meta.get('profile', '')} · prereg v{meta.get('prereg_version', '')}", "",
         f"Metric `{an['metric']}` · treat M={an['treat_M']} vs ctrl M={an['ctrl_M']} · "
         f"confirmatory seeds {an['confirmatory_seeds']} (tuning seed {an['tuning_seeds']} excluded) · "
         f"paired users {an['n_users_paired']}", "",
         f"Preregistration sha256 `{meta.get('prereg_sha256', '')[:16]}…` · selection split "
         f"`{meta.get('selection_split')}` · sigma policy `{meta.get('sigma_policy')}`", "",
         "All gating intervals are **two-level**: SE² = SE²_user + Var_seed/S, critical value t_{S−1}.", ""]
    d0 = an["D0"]
    if d0.get("available"):
        L += ["## D0 — tuning seed (descriptive, NOT evidence)", "",
              f"Δ = {_f(d0['delta'])} [{_f(d0['lo'])}, {_f(d0['hi'])}] (user-level CI only)", ""]
    c1 = an["C1"]
    L += ["## C1 — primary contrast", "", f"- Δ: {_tl(c1)}",
          f"- sign consistent across seeds: {c1['sign_consistent']}", "",
          "## D1 — capacity by M (only the primary contrast is inferential)", "",
          "| M | seeds | mean | per-seed means | Δ vs ctrl | user-level CI | cold mean |",
          "|---|---|---|---|---|---|---|"]
    for r in an["D1"]:
        ps = ", ".join(_f(v) for v in r["per_seed_mean"])
        if r["M"] == an["ctrl_M"]:
            L.append(f"| {r['M']} (ctrl) | {r['seeds']} | {_f(r['mean'])} | {ps} | — | — | {_f(r.get('cold_mean'))} |")
        else:
            L.append(f"| {r['M']} | {r['seeds']} | {_f(r['mean'])} | {ps} | {_f(r['delta_vs_ctrl'])} | "
                     f"[{_f(r['delta_lo_user'])}, {_f(r['delta_hi_user'])}] | {_f(r.get('cold_mean'))} |")
    L += ["", "## D2-A — λ̂ quintiles (1 = most diverse), descriptive", "",
          "| Q | N users | N test events | λ̂ range | Δ | user-level CI |", "|---|---|---|---|---|---|"]
    for r in an["D2A"]:
        L.append(f"| {r['lambda_quintile']} | {r['N_users']} | {r['N_test_events']} | "
                 f"{_f(r['lambda_min'],3)}–{_f(r['lambda_max'],3)} | {_f(r['delta'])} | "
                 f"[{_f(r['delta_lo'])}, {_f(r['delta_hi'])}] |")
    b = an["D2B"]
    L += ["", "## D2-B — β(λ̂), two-level", ""]
    for k in ("lambda", "lambda+logn", "lambda+logn+tail_test", "lambda+logn+tail_train"):
        L.append(f"- `{k}`: {_tl(b[k])}")
    L.append(f"- corr(λ̂, log n) = {_f(b['corr_lambda_logn'],3)}")
    d = an["D2C"]
    L += ["", "## D2-C — within history-length strata", "",
          "| stratum | N | mean n | β_j | user CI | R_j | DEFF_j | β_j/R_j | reliable |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s in d["strata"]:
        L.append(f"| {s['stratum']} | {s['N']} | {_f(s['mean_n_train'],1)} | {_f(s['beta'])} | "
                 f"[{_f(s['lo'])}, {_f(s['hi'])}] | {_f(s['R'],3)} | {_f(s['deff_median'],2)} | "
                 f"{_f(s['ratio_beta_over_R'])} | {'yes' if s['reliable'] else 'no'} |")
    L += ["", f"- stratum-FE slope: {_tl(d['fe_combined'])}",
          f"- β/R over reliable strata: mean {_f(d['ratio_mean'])}, CV {_f(d['ratio_cv'],3)}; "
          f"corr(β_j, R_j) = {_f(d['corr_beta_R'],3)}"]
    iv = an["D2D"]
    L += ["", "## D2-D — split-half IV over sessions (robustness only)", ""]
    if iv.get("ok"):
        for k in ("A_instrumented_by_B", "B_instrumented_by_A"):
            r = iv[k]
            L.append(f"- {k}: β={_f(r['beta'])} [{_f(r['lo'])}, {_f(r['hi'])}], "
                     f"first-stage F={_f(r['first_stage_F'],1)}, n={r['n']}")
    else:
        L.append(f"- not estimable (usable users: {iv['n_usable']})")
    L += ["", "## Robustness (never gating)", ""]
    for key, label in (("R1_logdt", "log1p(clipped days) time — NOT raw Δt"),
                       ("R2_popularity_negatives", "popularity-sampled negatives")):
        r = an.get(key, {})
        L.append(f"- {label}: " + (_tl(r) + f" · sign consistent {r['sign_consistent']}"
                                    if r.get("available") else "not run"))
    L += [f"- D17 max mean pairwise cos over analysed runs: {_f(an.get('D17_max_mean_cos'), 3)} "
          "(gated; detects cone collapse only — see run summaries for effective rank, mean |cos|, quantiles)",
          "", "## Acceptance", "", "| condition | pass |", "|---|---|"]
    for k, v in acc["conditions"].items():
        L.append(f"| {k} | {'✅' if v else '❌'} |")
    L += ["", "Procedural checks: " + ", ".join(f"{k}={'✅' if v else '❌'}" for k, v in acc["procedural"].items()),
          "", f"**Verdict: `{acc['verdict']}`**", ""]
    return "\n".join(L)

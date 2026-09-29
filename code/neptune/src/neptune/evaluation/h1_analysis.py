"""The H1 analysis (preregistration v1.2).

Two rules changed after external review, both about what counts as evidence:

1. Seed 0 is the TUNING seed.  h*(M), sigma*(M) were chosen on its validation
   score, so its test score carries winner's-curse bias.  It is reported as a
   descriptive row (D0) and never enters C1-C5.

2. Inference is two-level.  A CI over thousands of users measures user-sampling
   noise only; it can be tiny while the effect flips sign between training
   seeds (pseudo-replication at the level of trained models).  For every gating
   estimate theta:
       theta      = estimate on the seed-averaged per-user difference
       SE_total^2 = SE_user^2 (HC3 / sd/sqrt(n))  +  Var_s(theta^(s)) / S
       CI         = theta +/- t_{S-1, 1-alpha/2} * SE_total
   with theta^(s) the same estimate on seed s alone.  Seed s of the treatment
   arm is paired with seed s of the control arm: a COMMON-RANDOM-NUMBERS design.
   The same seed gives both arms an identical encoder / potential / satiation
   initialization, the same batch order and the same negative-sample stream (the
   M=1 prior particle is even the first of the M=16 prior particles), so
   Delta^(s) is a paired difference and Var_s(Delta^(s)) is the variance of that
   paired difference.  t with S-1 df is deliberately conservative.  C1 additionally requires sign consistency: every confirmatory
   seed's mean Delta must be > 0.

Tables
  D0    tuning seed, descriptive only
  D1    capacity per M (primary contrast two-level; other arms descriptive)
  D2-A  lambda_hat quintiles (descriptive)
  D2-B  Delta ~ lambda_hat [+ log n] [+ tail]            -> C2, C3, C4
  D2-C  within-length strata, reliability, FE slope       -> C5
  D2-D  session split-half IV (robustness only)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from ..semantics.reliability import reliability_by_stratum
from .stats import mean_ci, ols_hc3, tsls


# ---------------------------------------------------------------- plumbing

def per_user_seed(results: pd.DataFrame, metric: str, M: int, seeds) -> pd.DataFrame:
    """[users x seeds] for arm M; users missing from any seed are dropped."""
    sub = results[(results["M"] == M) & (results["seed"].isin(seeds))]
    piv = sub.pivot_table(index="u", columns="seed", values=metric, aggfunc="first")
    return piv.reindex(columns=list(seeds)).dropna()


def two_level(est: float, se_user: float, per_seed, alpha: float) -> dict:
    per_seed = [float(x) for x in per_seed]
    S = len(per_seed)
    if S < 2 or not np.isfinite(se_user):
        return {"est": est, "se_user": se_user, "sd_seed": float("nan"), "se_total": float("nan"),
                "df": S - 1, "lo": float("nan"), "hi": float("nan"), "per_seed": per_seed}
    sd_seed = float(np.std(per_seed, ddof=1))
    se_total = float(np.sqrt(se_user ** 2 + sd_seed ** 2 / S))
    crit = float(student_t.ppf(1 - alpha / 2, S - 1))
    return {"est": float(est), "se_user": float(se_user), "sd_seed": sd_seed, "se_total": se_total,
            "df": S - 1, "lo": float(est - crit * se_total), "hi": float(est + crit * se_total),
            "per_seed": per_seed}


def primary_contrast(results, metric, treat_M, ctrl_M, seeds, alpha) -> tuple:
    """Per-user per-seed Delta [users x seeds] (common users) and the C1 summary."""
    T = per_user_seed(results, metric, treat_M, seeds)
    C = per_user_seed(results, metric, ctrl_M, seeds)
    users = T.index.intersection(C.index)
    D = T.loc[users] - C.loc[users]
    dbar = D.mean(axis=1)
    m = mean_ci(dbar, alpha)
    per_seed = [float(D[s].mean()) for s in seeds]
    c1 = two_level(m["mean"], m["se"], per_seed, alpha)
    c1.update({"n_users": int(len(dbar)), "seeds": list(seeds),
               "sign_consistent": bool(all(x > 0 for x in per_seed))})
    return D, c1


# ------------------------------------------------------------------ tables

def d0_tuning(results, metric, treat_M, ctrl_M, tuning_seeds, alpha) -> dict:
    T = per_user_seed(results, metric, treat_M, tuning_seeds)
    C = per_user_seed(results, metric, ctrl_M, tuning_seeds)
    users = T.index.intersection(C.index)
    if len(users) == 0:
        return {"available": False}
    ci = mean_ci((T.loc[users] - C.loc[users]).mean(axis=1), alpha)
    return {"available": True, "seeds": list(tuning_seeds), "delta": ci["mean"], "lo": ci["lo"],
            "hi": ci["hi"], "note": "tuning seed: selected on its validation score; descriptive only"}


def d1_table(results, metric, seeds_by_arm: dict, ctrl_M: int, alpha: float, c1: dict,
             treat_M: int) -> list:
    """Each arm is compared with the control averaged over the SAME seeds as the
    arm (matched common random numbers) -- M=4 with seeds {1,2} is compared with
    M=1 on seeds {1,2}, not {1..5}."""
    cold_col = f"{metric}_cold"
    rows = []
    for M, seeds in sorted(seeds_by_arm.items()):
        missing = set(seeds) - set(seeds_by_arm[ctrl_M])
        if missing:
            raise ValueError(f"control arm lacks seeds {sorted(missing)} needed to match M={M}")
        ctrl = per_user_seed(results, metric, ctrl_M, seeds).mean(axis=1)
        arm = per_user_seed(results, metric, M, seeds)
        users = arm.index.intersection(ctrl.index)
        row = {"M": int(M), "seeds": list(seeds), "n_users": int(len(users)),
               "mean": float(arm.mean(axis=1).mean()),
               "per_seed_mean": [float(arm[s].mean()) for s in seeds]}
        if M != ctrl_M:
            ci = mean_ci(arm.loc[users].mean(axis=1) - ctrl.loc[users], alpha)
            row.update({"delta_vs_ctrl": ci["mean"], "delta_lo_user": ci["lo"], "delta_hi_user": ci["hi"]})
        if M == treat_M:
            row["two_level"] = {k: c1[k] for k in ("est", "lo", "hi", "se_total", "sd_seed", "df",
                                                   "sign_consistent")}
        if cold_col in results:
            cold = per_user_seed(results, cold_col, M, seeds)
            row["cold_mean"] = float(cold.mean(axis=1).mean()) if len(cold) else float("nan")
        rows.append(row)
    return rows


def delta_frame(D: pd.DataFrame, cov: pd.DataFrame) -> tuple:
    df = pd.DataFrame({"u": D.index.to_numpy(), "delta": D.mean(axis=1).to_numpy()})
    seed_cols = []
    for s in D.columns:
        col = f"delta_s{s}"
        df[col] = D[s].to_numpy()
        seed_cols.append(col)
    return df.merge(cov, on="u", how="inner"), seed_cols


def d2a(df: pd.DataFrame, alpha: float) -> list:
    rows = []
    for q, g in df.groupby("lambda_quintile"):
        ci = mean_ci(g["delta"], alpha)
        rows.append({"lambda_quintile": int(q), "N_users": int(len(g)),
                     "N_test_events": int(g["n_test"].sum()),
                     "lambda_min": float(g["lambda_hat"].min()), "lambda_max": float(g["lambda_hat"].max()),
                     "delta": ci["mean"], "delta_lo": ci["lo"], "delta_hi": ci["hi"]})
    return rows


D2B_MODELS = {
    "lambda": ["lambda_hat"],
    "lambda+logn": ["lambda_hat", "log_n"],
    "lambda+logn+tail_test": ["lambda_hat", "log_n", "tail_test"],
    "lambda+logn+tail_train": ["lambda_hat", "log_n", "tail_train"],
}


def _slope_two_level(df, cols, seed_cols, alpha, extra=None) -> dict:
    extra = extra or {}
    sub = df.dropna(subset=["delta"] + cols)
    fit = ols_hc3(sub["delta"], {**{c: sub[c] for c in cols}, **extra}, alpha)
    if not fit.get("ok"):
        return {"ok": False}
    c = fit["coef"]["lambda_hat"]
    per = []
    for sc in seed_cols:
        f = ols_hc3(sub[sc], {**{k: sub[k] for k in cols}, **extra}, alpha)
        per.append(f["coef"]["lambda_hat"]["beta"] if f.get("ok") else float("nan"))
    out = two_level(c["beta"], c["se"], per, alpha)
    out.update({"ok": True, "n": fit["n"], "user_lo": c["lo"], "user_hi": c["hi"]})
    return out


def d2b(df, seed_cols, alpha) -> dict:
    out = {name: _slope_two_level(df, cols, seed_cols, alpha) for name, cols in D2B_MODELS.items()}
    a, b = df["lambda_hat"].to_numpy(float), df["log_n"].to_numpy(float)
    out["corr_lambda_logn"] = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    return out


def d2c(df: pd.DataFrame, seed_cols, alpha: float, r_min: float) -> dict:
    rel = reliability_by_stratum(df["lambda_hat"].to_numpy(), df["tau2_block"].to_numpy(),
                                 df["length_stratum"].to_numpy(), df["deff"].to_numpy())
    strata = []
    for _, r in rel.iterrows():
        g = df[df["length_stratum"] == r["stratum"]]
        fit = ols_hc3(g["delta"], {"lambda_hat": g["lambda_hat"]}, alpha)
        c = fit["coef"]["lambda_hat"] if fit.get("ok") else {"beta": np.nan, "se": np.nan, "lo": np.nan, "hi": np.nan}
        R = float(r["R"])
        strata.append({
            "stratum": int(r["stratum"]), "N": int(r["n"]), "mean_n_train": float(g["n_train"].mean()),
            "beta": c["beta"], "se": c["se"], "lo": c["lo"], "hi": c["hi"], "R": R,
            "var_lambda_obs": float(r["var_lambda_obs"]), "mean_tau2": float(r["mean_tau2"]),
            "deff_median": float(r.get("deff_median", np.nan)),
            "ratio_beta_over_R": c["beta"] / R if np.isfinite(R) and R > 0 else float("nan"),
            "reliable": bool(np.isfinite(R) and R >= r_min),
        })
    dummies = {f"stratum_{j}": (df["length_stratum"] == j).astype(float)
               for j in sorted(df["length_stratum"].unique())[1:]}
    fe = _slope_two_level(df, ["lambda_hat"], seed_cols, alpha, extra=dummies)
    rel_rows = [s for s in strata if s["reliable"] and np.isfinite(s["ratio_beta_over_R"])]
    ratios = np.array([s["ratio_beta_over_R"] for s in rel_rows])
    betas = np.array([s["beta"] for s in strata])
    Rs = np.array([s["R"] for s in strata])
    ok = np.isfinite(betas) & np.isfinite(Rs)
    return {
        "strata": strata, "fe_combined": fe, "n_reliable": int(len(rel_rows)),
        "ratio_mean": float(ratios.mean()) if len(ratios) else float("nan"),
        "ratio_cv": float(ratios.std(ddof=1) / abs(ratios.mean())) if len(ratios) >= 2 and ratios.mean() != 0 else float("nan"),
        "corr_beta_R": float(np.corrcoef(betas[ok], Rs[ok])[0, 1]) if ok.sum() >= 3 else float("nan"),
    }


def d2d(df: pd.DataFrame, alpha: float, n_boot: int, seed: int) -> dict:
    sub = df.dropna(subset=["lambda_A", "lambda_B", "tail_test"])
    out = {"n_usable": int(len(sub)), "note": "robustness only; between-session persistence may remain"}
    if len(sub) < 20:
        out["ok"] = False
        return out
    ctrl = {"log_n": sub["log_n"], "tail_test": sub["tail_test"]}
    out["A_instrumented_by_B"] = tsls(sub["delta"], sub["lambda_A"], sub["lambda_B"], ctrl,
                                      n_boot=n_boot, seed=seed, alpha=alpha)
    out["B_instrumented_by_A"] = tsls(sub["delta"], sub["lambda_B"], sub["lambda_A"], ctrl,
                                      n_boot=n_boot, seed=seed + 1, alpha=alpha)
    out["ok"] = True
    return out


def run_h1_analysis(results: pd.DataFrame, cov: pd.DataFrame, *, metric: str, seeds_by_arm: dict,
                    tuning_seeds, treat_M: int, ctrl_M: int, alpha: float, r_min: float,
                    n_boot_iv: int, seed: int = 0) -> dict:
    conf = results[~results["seed"].isin(tuning_seeds)]
    primary_seeds = seeds_by_arm[treat_M]
    if list(primary_seeds) != list(seeds_by_arm[ctrl_M]):
        raise ValueError("treatment and control arms must share their confirmatory seeds")
    if set(primary_seeds) & set(tuning_seeds):
        raise ValueError("a tuning seed leaked into the confirmatory set")
    D, c1 = primary_contrast(conf, metric, treat_M, ctrl_M, primary_seeds, alpha)
    df, seed_cols = delta_frame(D, cov)
    return {
        "metric": metric, "treat_M": treat_M, "ctrl_M": ctrl_M,
        "confirmatory_seeds": list(primary_seeds), "tuning_seeds": list(tuning_seeds),
        "n_users_paired": int(len(df)),
        "D0": d0_tuning(results, metric, treat_M, ctrl_M, tuning_seeds, alpha),
        "C1": c1,
        "D1": d1_table(conf, metric, seeds_by_arm, ctrl_M, alpha, c1, treat_M),
        "D2A": d2a(df, alpha),
        "D2B": d2b(df, seed_cols, alpha),
        "D2C": d2c(df, seed_cols, alpha, r_min),
        "D2D": d2d(df, alpha, n_boot_iv, seed),
    }

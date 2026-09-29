"""Small, dependency-free inference toolkit for the H1 analysis.

  ols_hc3        OLS with HC3 heteroskedasticity-robust standard errors
  tsls           just-identified 2SLS, user-bootstrap CI, robust first-stage F
  mean_ci        mean of a per-user paired difference with a normal CI
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np


def z_crit(alpha: float) -> float:
    return NormalDist().inv_cdf(1.0 - alpha / 2.0)


def _design(cols: dict, n: int):
    names = ["const"] + list(cols)
    X = np.column_stack([np.ones(n)] + [np.asarray(cols[k], dtype=np.float64) for k in cols])
    return X, names


def ols_hc3(y, cols: dict, alpha: float = 0.05) -> dict:
    y = np.asarray(y, dtype=np.float64)
    X, names = _design(cols, len(y))
    n, k = X.shape
    if n <= k + 1:
        return {"n": int(n), "ok": False}
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    lev = np.einsum("ij,jk,ik->i", X, XtX_inv, X).clip(max=1 - 1e-8)
    meat = (X * (e / (1.0 - lev))[:, None] ** 2).T @ X
    V = XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    zc = z_crit(alpha)
    coef = {nm: {"beta": float(b), "se": float(s), "lo": float(b - zc * s), "hi": float(b + zc * s),
                 "z": float(b / s) if s > 0 else float("nan")}
            for nm, b, s in zip(names, beta, se)}
    return {"n": int(n), "ok": True, "coef": coef}


def _tsls_point(y, x, z, C):
    """y = b x + C g + e, x instrumented by z (C includes the constant)."""
    Z = np.column_stack([z, C])
    xhat = Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    return np.linalg.lstsq(np.column_stack([xhat, C]), y, rcond=None)[0][0]


def tsls(y, x_endog, z_instr, controls: dict, *, n_boot: int, seed: int, alpha: float = 0.05) -> dict:
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x_endog, dtype=np.float64)
    z = np.asarray(z_instr, dtype=np.float64)
    n = len(y)
    C, _ = _design(controls, n)
    beta = float(_tsls_point(y, x, z, C))
    first = ols_hc3(x, {"instrument": z, **controls}, alpha)
    t = first["coef"]["instrument"]["z"] if first.get("ok") else float("nan")
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            boots.append(_tsls_point(y[idx], x[idx], z[idx], C[idx]))
        except np.linalg.LinAlgError:
            continue
    boots = np.asarray(boots)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2]) if len(boots) else (np.nan, np.nan)
    return {"n": int(n), "beta": beta, "lo": float(lo), "hi": float(hi),
            "se_boot": float(boots.std(ddof=1)) if len(boots) > 1 else float("nan"),
            "first_stage_F": float(t * t), "n_boot_ok": int(len(boots))}


def mean_ci(x, alpha: float = 0.05) -> dict:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return {"n": int(n), "mean": float(x.mean()) if n else float("nan"),
                "se": float("nan"), "lo": float("nan"), "hi": float("nan")}
    m, se = float(x.mean()), float(x.std(ddof=1) / np.sqrt(n))
    zc = z_crit(alpha)
    return {"n": int(n), "mean": m, "se": se, "lo": m - zc * se, "hi": m + zc * se}

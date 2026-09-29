"""Unbiased Simpson collision probability (amendment A1).

    lambda_hat = sum_c n_c (n_c - 1) / ( n (n - 1) )

Under multinomial sampling E[n_c(n_c-1)] = n(n-1) p_c^2, so lambda_hat is
unbiased for lambda = sum_c p_c^2 at every n.  That removes the mechanical
ceiling  H_plugin <= log min(n, K)  which made Shannon entropy partly a measure
of history LENGTH.

Two limits worth stating precisely:
  * unbiasedness holds under multinomial (i.i.d.) sampling; real consumption
    is session-clustered, so this is a claim about the estimator, not the data;
  * unbiased mean does NOT mean equal precision: Var(lambda_hat) still shrinks
    with n, which is why A1.1 (reliability) exists.

Convention: HIGH lambda_hat = concentrated history; LOW = diverse / multimodal.
Do not use 1/lambda_hat: the reciprocal of an unbiased estimator is biased.
"""

from __future__ import annotations

import numpy as np


def unbiased_collision(counts):
    c = np.asarray(counts, dtype=np.float64)
    squeeze = c.ndim == 1
    if squeeze:
        c = c[None, :]
    if np.any(c < 0):
        raise ValueError("counts must be non-negative")
    n = c.sum(axis=1)
    if np.any(n < 2):
        raise ValueError("unbiased Simpson collision is undefined for n < 2 (division by n(n-1))")
    lam = (c * (c - 1.0)).sum(axis=1) / (n * (n - 1.0))
    if np.any(lam < -1e-12) or np.any(lam > 1.0 + 1e-12):
        raise ValueError("collision estimate outside [0, 1]: corrupted counts")
    lam = np.clip(lam, 0.0, 1.0)
    return float(lam[0]) if squeeze else lam

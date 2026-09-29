"""Split-half over alternating SESSIONS (amendment A1.2).

Alternating EVENTS would put half of every binge session in each half, so both
halves would share that session's cluster -- maximizing Cov(err_A, err_B),
which is exactly the correlation that makes an IV fail.  Alternating whole
sessions breaks within-session dependence while keeping both halves spread over
the same calendar span (a chronological first-half / second-half split would
instead confound measurement with preference drift).

Positive between-session persistence can remain, so the IV built on this is a
robustness diagnostic and is labelled as such -- not a measurement-error-free
estimate.
"""

from __future__ import annotations

import numpy as np

from .simpson import unbiased_collision


def session_split_half(session_counts: np.ndarray):
    sc = np.asarray(session_counts, dtype=np.int64)
    a = sc[0::2].sum(axis=0)
    b = sc[1::2].sum(axis=0)
    na, nb = int(a.sum()), int(b.sum())
    lam_a = unbiased_collision(a) if na >= 2 else float("nan")
    lam_b = unbiased_collision(b) if nb >= 2 else float("nan")
    return lam_a, lam_b, na, nb

"""Session assignment.

A session is a maximal run of a user's events with consecutive gaps no larger
than `gap_seconds` (preregistered: 1800 s).  Sessions are always defined on REAL
timestamps, even when the model runs on ordinal time: they exist to describe the
dependence structure of the data (bulk-rating binges), which is what the
session-block bootstrap and the session split-half need.
"""

from __future__ import annotations

import numpy as np


def assign_sessions(user: np.ndarray, t: np.ndarray, gap_seconds: float) -> np.ndarray:
    """Per-user local session index (0, 1, 2, ...).  Input must be sorted by (user, t)."""
    n = len(user)
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    user_change = np.r_[True, user[1:] != user[:-1]]
    new_session = user_change.copy()
    new_session[1:] |= (t[1:] - t[:-1]) > gap_seconds
    gid = np.cumsum(new_session) - 1
    start_idx = np.maximum.accumulate(np.where(user_change, np.arange(n), 0))
    return (gid - gid[start_idx]).astype(np.int32)

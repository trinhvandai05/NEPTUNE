"""Load MovieLens positive events restricted to the genome-covered catalog."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_positive_events(raw_dir, movie_ids: np.ndarray, threshold: float,
                         return_first_seen: bool = False):
    """Return columns [userId:int64, item:int32, t:float64] for ratings >= threshold
    on items that have genome features.

    With return_first_seen=True also returns first_seen[N]: the earliest timestamp
    of ANY rating (any value, any user) of each genome item, np.inf if never
    rated.  Cold-item status is defined from this corpus-wide quantity, BEFORE
    thresholding and cohort sampling: an item rated before T_cut only by users
    later excluded from the cohort is still not cold (spec: "first interaction
    falls after T_cut").

    Implicit feedback by thresholding is the MovieLens convention; it is applied
    identically to every arm, so it cannot favour one representation over another.
    Items without genome have no features, hence no coordinates for e_phi, and are
    dropped here -- the usable catalog is the genome-covered subset.
    """
    raw_dir = Path(raw_dir)
    r = pd.read_csv(raw_dir / "ratings.csv",
                    dtype={"userId": np.int64, "movieId": np.int64,
                           "rating": np.float32, "timestamp": np.int64})
    mid = r["movieId"].to_numpy()
    pos = np.searchsorted(movie_ids, mid)
    pos_clip = np.minimum(pos, len(movie_ids) - 1)
    has_genome = movie_ids[pos_clip] == mid
    keep = has_genome & (r["rating"].to_numpy() >= threshold)
    first_seen = np.full(len(movie_ids), np.inf)
    np.minimum.at(first_seen, pos_clip[has_genome], r["timestamp"].to_numpy()[has_genome].astype(np.float64))
    events = pd.DataFrame({
        "userId": r["userId"].to_numpy()[keep],
        "item": pos_clip[keep].astype(np.int32),
        "t": r["timestamp"].to_numpy()[keep].astype(np.float64),
    })
    return (events, first_seen) if return_first_seen else events

"""Synthetic data in the exact MovieLens-25M file format.

Writes ratings.csv, movies.csv and genome-scores.csv with the real column names
and dtypes, so the production `prepare` path runs unchanged.  Structure is
planted, not random: items belong to latent clusters, each user has 1-4
interests (so lambda_hat genuinely varies), consumption comes in sessions
(so the block bootstrap has something to find), some items are released late
(cold items), and some movies have no genome row (exercising the filter).

This exists for tests and for `scripts/smoke_pipeline.py`.  It says nothing
about MovieLens; no result obtained on it is evidence for or against H1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

GENRES = ["Action", "Adventure", "Comedy", "Crime", "Drama", "Horror", "Romance",
          "Sci-Fi", "Thriller", "Animation"]


def write_synthetic_ml25m(out_dir, *, n_users: int = 240, n_items: int = 220, n_tags: int = 48,
                          n_clusters: int = 6, n_no_genome: int = 12, seed: int = 0,
                          span_days: float = 500.0) -> Path:
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0, span = 1_300_000_000.0, span_days * 86400.0

    n_total = n_items + n_no_genome
    movie_ids = np.arange(1, n_total + 1) * 7          # sparse ids, like the real file
    cluster = rng.integers(0, n_clusters, n_total)
    centers = rng.normal(0, 2.0, (n_clusters, n_tags))
    relevance = 1 / (1 + np.exp(-(centers[cluster[:n_items]] + rng.normal(0, 0.6, (n_items, n_tags)))))
    release = np.where(rng.random(n_total) < 0.08, t0 + span * rng.uniform(0.86, 0.95, n_total), t0)

    cg = [rng.choice(len(GENRES), 2, replace=False) for _ in range(n_clusters)]
    titles, genres = [], []
    for k, mid in enumerate(movie_ids):
        year = int(rng.integers(1950, 2020))
        titles.append(f"Synthetic Movie {mid} ({year})")
        g = set(cg[cluster[k]].tolist())
        if rng.random() < 0.3:
            g.add(int(rng.integers(len(GENRES))))
        genres.append("|".join(GENRES[j] for j in sorted(g)))
    pd.DataFrame({"movieId": movie_ids, "title": titles, "genres": genres}).to_csv(out / "movies.csv", index=False)

    gm = np.repeat(movie_ids[:n_items], n_tags)
    gt = np.tile(np.arange(1, n_tags + 1), n_items)
    pd.DataFrame({"movieId": gm, "tagId": gt, "relevance": relevance.ravel().round(5)}).to_csv(
        out / "genome-scores.csv", index=False)

    by_cluster = [np.flatnonzero(cluster == c) for c in range(n_clusters)]
    rows = []
    for u in range(1, n_users + 1):
        k = int(rng.choice([1, 1, 2, 2, 3, 4]))
        interests = rng.choice(n_clusters, k, replace=False)
        start = t0 + span * rng.uniform(0.0, 0.5)
        end = t0 + span * (rng.uniform(0.97, 1.0) if rng.random() < 0.85 else rng.uniform(0.6, 0.8))
        n_sess = int(rng.integers(6, 22))
        sess_start = np.sort(rng.uniform(start, end, n_sess))
        used = set()
        for s0 in sess_start:
            focus = rng.choice(interests)                  # sessions are topically coherent
            ts = s0
            for _ in range(int(rng.integers(1, 7))):
                ts += rng.uniform(30, 600)
                in_interest = rng.random() < 0.85
                pool = by_cluster[focus] if in_interest else np.arange(n_total)
                pool = [j for j in pool if j not in used and release[j] <= ts]
                if not pool:
                    continue
                j = int(rng.choice(pool))
                used.add(j)
                rating = float(rng.choice([4.0, 4.5, 5.0])) if in_interest else float(rng.choice([1.0, 2.0, 3.0, 4.0]))
                rows.append((u, int(movie_ids[j]), rating, int(ts)))
    r = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])
    r.sort_values(["userId", "timestamp"], kind="mergesort").to_csv(out / "ratings.csv", index=False)
    return out

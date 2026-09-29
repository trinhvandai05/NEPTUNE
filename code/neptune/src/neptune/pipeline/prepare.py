"""I00_DATA: raw MovieLens-25M -> frozen artifacts."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..common import sha256_file, write_json_atomic
from ..config import ConfigError
from ..provenance import implementation_sha256
from ..content.raw_features import build_raw_features
from ..data.ml25m import load_positive_events
from ..data.split import TEST, TRAIN, temporal_split


# Bump whenever the CONTENT of the artifact changes for the same data_cfg
# (1.2: corpus-wide, any-rating cold-item definition).  Claim runs refuse any other.
DATA_SCHEMA = "neptune-data/1.2.3"      # 1.2.3: payload files hashed in the manifest


def cold_items_from_first_seen(first_seen: np.ndarray, t_cut: float) -> np.ndarray:
    """Cold = first rating of ANY value by ANY user at/after T_cut (never-rated
    items count as cold: the system has no interaction with them either)."""
    return first_seen >= t_cut


def prepare_dataset(raw_dir, out_dir, cfg, seed: int = 0) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d = cfg.data

    movie_ids, X, finfo = build_raw_features(raw_dir, d.genome_components, seed)
    ev, first_seen = load_positive_events(raw_dir, movie_ids, d.positive_threshold, return_first_seen=True)
    sp = temporal_split(ev, n_users=d.n_users, train_frac=d.train_frac, val_frac=d.val_frac,
                        min_train_events=d.min_train_events, min_test_events=d.min_test_events,
                        session_gap_seconds=d.session_gap_seconds, seed=seed)
    events, n_items = sp.events, len(movie_ids)
    if sp.attrition["users_sampled"] < d.n_users:
        # Not a usable artifact: no manifest.json is written, so nothing can run on it.
        write_json_atomic(out / "cohort_shortfall.json", {"requested_n_users": d.n_users,
                                                           "attrition": sp.attrition})
        raise ConfigError(
            f"cohort shortfall: {sp.attrition['users_sampled']} evaluable users < requested {d.n_users} "
            f"(see {out / 'cohort_shortfall.json'}). Amend data.n_users transparently; do not run on "
            "a silently smaller cohort.")

    popularity = np.bincount(events.loc[events["split"] == TRAIN, "i"], minlength=n_items).astype(np.float32)
    cold = cold_items_from_first_seen(first_seen, sp.t_cut)
    warm = ~cold                  # negative-sampling support: seen by the system before T_cut
    test_items = np.unique(events.loc[events["split"] == TEST, "i"].to_numpy())
    n_cold_test = int(cold[test_items].sum())

    np.save(out / "item_features.npy", X)
    np.save(out / "movie_ids.npy", movie_ids)
    np.save(out / "popularity.npy", popularity)
    np.save(out / "cold_items.npy", cold)
    np.save(out / "warm_items.npy", warm)
    np.save(out / "user_ids.npy", sp.user_ids)
    events.to_parquet(out / "events.parquet", index=False)

    payload = ("item_features.npy", "movie_ids.npy", "popularity.npy", "cold_items.npy",
               "warm_items.npy", "user_ids.npy", "events.parquet")
    counts = events["split"].value_counts().to_dict()
    manifest = {
        "schema": DATA_SCHEMA, "prereg_sha256": cfg.prereg_sha256,
        "implementation_sha256": implementation_sha256(),
        "data_cfg": asdict(d), "seed": seed,
        "n_users": int(events["u"].nunique()), "n_items": int(n_items), "d_x": int(X.shape[1]),
        "n_events": int(len(events)),
        "split_counts": {"train": int(counts.get(0, 0)), "val": int(counts.get(1, 0)), "test": int(counts.get(2, 0))},
        "t_cut": sp.t_cut, "t_val": sp.t_val, "attrition": sp.attrition, "features": finfo,
        "n_warm_items": int(warm.sum()), "n_cold_items": int(cold.sum()),
        "n_cold_items_in_test": n_cold_test,
        "cold_arm_usable": bool(n_cold_test >= d.cold_min_items),
        "negatives_restricted_to_warm": True,
        "files": {name: sha256_file(out / name) for name in payload},
        "cold_definition": "first rating of any value by any user (full corpus, pre-cohort) >= T_cut",
    }
    write_json_atomic(out / "manifest.json", manifest)
    return manifest

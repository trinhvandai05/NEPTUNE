"""I01_COVARIATES: model-independent H1 covariates, frozen before any training."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..common import sha256_file, write_json_atomic
from ..provenance import implementation_sha256
from ..semantics.clustering import fit_clusters
from ..semantics.covariates import compute_user_covariates
from .artifacts import load_artifacts


SEMANTICS_SCHEMA = "neptune-semantics/1.2.3"   # 1.2.3: payload hashed, linked to the data manifest


def build_covariates(data_dir, cfg, seed: int = 0):
    data_dir = Path(data_dir)
    art = load_artifacts(data_dir)
    s = cfg.semantics
    labels, centroids, cinfo = fit_clusters(art.X, s.n_clusters, s.kmeans_seed)
    cov, head, info = compute_user_covariates(
        art.events, labels, art.n_items, head_fraction=s.head_fraction,
        n_boot=s.n_boot_reliability, n_strata=s.n_strata, seed=seed)
    np.save(data_dir / "item_cluster.npy", labels)
    np.save(data_dir / "cluster_centroids.npy", centroids)
    np.save(data_dir / "head_items.npy", head)
    cov.to_parquet(data_dir / "covariates.parquet", index=False)
    manifest = {"schema": SEMANTICS_SCHEMA, "prereg_sha256": cfg.prereg_sha256,
                "implementation_sha256": implementation_sha256(),
                "data_manifest_sha256": sha256_file(data_dir / "manifest.json"),
                "semantics_cfg": vars(s).copy(),
                "clustering": cinfo, **info,
                "files": {name: sha256_file(data_dir / name) for name in
                          ("covariates.parquet", "item_cluster.npy", "cluster_centroids.npy", "head_items.npy")},
                "note": "computed from raw content + train split only; never reads a model"}
    write_json_atomic(data_dir / "semantics_manifest.json", manifest)
    return cov, manifest

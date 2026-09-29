"""k-means on RAW content features -- run once, before any training."""

from __future__ import annotations

import numpy as np


def fit_clusters(X_raw: np.ndarray, n_clusters: int, seed: int):
    from sklearn.cluster import MiniBatchKMeans

    k = int(min(n_clusters, X_raw.shape[0]))
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=4096, n_init=5)
    km.fit(X_raw)
    return km.labels_.astype(np.int32), km.cluster_centers_.astype(np.float32), {
        "n_clusters_requested": int(n_clusters), "n_clusters_effective": k,
        "inertia": float(km.inertia_),
    }

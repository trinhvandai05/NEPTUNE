"""Raw item content features for MovieLens-25M.

    X_raw = standardize( [ PCA(tag-genome relevance) | genre multi-hot | year ] )

X_raw has two consumers that must stay separate:
  * the learned encoder e_phi (neptune.content.encoder), and
  * the model-independent clustering (neptune.semantics.clustering) that
    defines the H1 stratification variable.  The clustering sees X_raw only,
    never an encoder output.

Deviation from spec sec. 4: no title text embedding.  It would add a heavy
sentence-encoder dependency for information the 1128-tag genome already carries
far more densely.  Recorded in the data manifest as `title_embedding: false`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_genome(raw_dir: Path):
    g = pd.read_csv(raw_dir / "genome-scores.csv",
                    dtype={"movieId": np.int64, "tagId": np.int64, "relevance": np.float32})
    movies = np.sort(g["movieId"].unique())
    tags = np.sort(g["tagId"].unique())
    mat = np.zeros((len(movies), len(tags)), np.float32)
    mat[np.searchsorted(movies, g["movieId"].to_numpy()),
        np.searchsorted(tags, g["tagId"].to_numpy())] = g["relevance"].to_numpy()
    return movies, mat


def genre_year_block(raw_dir: Path, movie_ids: np.ndarray):
    mv = pd.read_csv(raw_dir / "movies.csv").set_index("movieId").reindex(movie_ids)
    genres = mv["genres"].fillna("").astype(str)
    names = sorted({x for s in genres for x in s.split("|") if x and x != "(no genres listed)"})
    col = {n: j for j, n in enumerate(names)}
    G = np.zeros((len(movie_ids), len(names)), np.float32)
    for r, s in enumerate(genres):
        for x in s.split("|"):
            j = col.get(x)
            if j is not None:
                G[r, j] = 1.0
    year = pd.to_numeric(mv["title"].fillna("").astype(str).str.extract(r"\((\d{4})\)\s*$")[0],
                         errors="coerce")
    fill = float(year.median()) if year.notna().any() else 1990.0
    year = year.fillna(fill).to_numpy(np.float32)
    return np.concatenate([G, ((year - 1990.0) / 30.0).reshape(-1, 1)], axis=1), names


def build_raw_features(raw_dir, genome_components: int, seed: int = 0):
    from sklearn.decomposition import PCA

    raw_dir = Path(raw_dir)
    movie_ids, mat = load_genome(raw_dir)
    n_comp = int(min(genome_components, mat.shape[0] - 1, mat.shape[1]))
    pca = PCA(n_components=n_comp, random_state=seed)
    reduced = pca.fit_transform(mat).astype(np.float32)
    side, genre_names = genre_year_block(raw_dir, movie_ids)
    X = np.concatenate([reduced, side], axis=1)
    X = ((X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-6)).astype(np.float32)
    info = {
        "n_genome_items": int(mat.shape[0]), "n_tags": int(mat.shape[1]),
        "genome_components_effective": n_comp,
        "genome_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "n_genres": len(genre_names), "d_x": int(X.shape[1]), "title_embedding": False,
    }
    return movie_ids, X, info

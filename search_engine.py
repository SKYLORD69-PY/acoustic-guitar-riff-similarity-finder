"""Vector similarity search engine.

Wraps a normalized feature matrix in a scikit-learn NearestNeighbors
index built with cosine distance, so any riff already in the library
-- or any new query vector produced by the same extraction pipeline --
can be compared against every other riff in one call.

Cosine distance is used because it measures the *angle* between
feature vectors rather than their magnitude: two riffs with the same
tonal profile but different recording levels should still register as
musically similar. The fitted index, the filenames it was trained on,
and the feature matrix itself are bundled into a single artifact so
they can never drift out of sync with one another.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

PathLike = Union[str, Path]

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)

DEFAULT_FEATURE_CSV = Path("data/features/feature_matrix.csv")
DEFAULT_INDEX_PATH = Path("data/features/search_index.joblib")
DEFAULT_N_NEIGHBORS = 5
FILENAME_COLUMN = "filename"


def load_feature_matrix(csv_path: PathLike) -> Tuple[List[str], np.ndarray]:
    """Load a normalized feature matrix produced by extract_features.py.

    Args:
        csv_path: Path to the CSV written by extract_features.py. Accepts
            a string or a Path -- this module is imported directly by
            recommend.py and app.py, where a path may arrive as either.

    Returns:
        Tuple of (filenames in row order, feature matrix as a 2D
        array with one row per file).

    Raises:
        FileNotFoundError: If `csv_path` does not exist.
        ValueError: If the CSV has no `filename` column or no rows.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found -- run extract_features.py first.")
    df = pd.read_csv(csv_path)
    if FILENAME_COLUMN not in df.columns:
        raise ValueError(f"Expected a '{FILENAME_COLUMN}' column in {csv_path}")
    if df.empty:
        raise ValueError(f"{csv_path} has no rows -- nothing to index.")
    filenames = df[FILENAME_COLUMN].tolist()
    feature_matrix = df.drop(columns=[FILENAME_COLUMN]).to_numpy(dtype=float)
    return filenames, feature_matrix


def build_search_index(feature_matrix: np.ndarray, n_neighbors: int) -> NearestNeighbors:
    """Fit a cosine-distance NearestNeighbors index over the feature matrix.

    Cosine distance (1 - cosine similarity) is used instead of the
    Euclidean default because it compares vectors by *direction*
    rather than magnitude -- the right notion of "similar" once every
    feature has already been standardized in extract_features.py.
    `algorithm="brute"` is set explicitly: scikit-learn's tree-based
    neighbor algorithms (ball_tree, kd_tree) don't support the cosine
    metric at all, and brute force is more than fast enough at the
    library sizes this project targets.

    Args:
        feature_matrix: 2D array of standardized features, one row
            per file.
        n_neighbors: Default number of neighbors to fit the index
            with; individual queries can still ask for a different k.

    Returns:
        Fitted NearestNeighbors index.
    """
    n_neighbors = min(n_neighbors, len(feature_matrix))
    model = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
    model.fit(feature_matrix)
    return model


def save_search_index(
    model: NearestNeighbors,
    filenames: List[str],
    feature_matrix: np.ndarray,
    index_path: PathLike,
) -> None:
    """Persist the fitted index alongside the data it was built from.

    The model, filenames, and feature matrix are bundled into one
    artifact so they can never fall out of sync -- loading a stale
    model against a regenerated feature matrix would silently return
    wrong or out-of-range neighbor results.

    Args:
        model: Fitted NearestNeighbors index.
        filenames: Filenames in the same row order the model was
            fitted on.
        feature_matrix: The feature matrix the model was fitted on.
        index_path: Destination path for the serialized artifact.
            Accepts a string or a Path.
    """
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "filenames": filenames, "feature_matrix": feature_matrix},
        index_path,
    )
    logger.info("Search index written to %s", index_path)


def load_search_index(index_path: PathLike) -> Tuple[NearestNeighbors, List[str], np.ndarray]:
    """Load a previously saved search index artifact.

    Args:
        index_path: Path to the artifact written by `save_search_index`.
            Accepts a string or a Path.

    Returns:
        Tuple of (fitted NearestNeighbors index, filenames, feature
        matrix), exactly as they were when saved.

    Raises:
        FileNotFoundError: If `index_path` does not exist.
    """
    index_path = Path(index_path)
    if not index_path.exists():
        raise FileNotFoundError(f"{index_path} not found -- run search_engine.py first to build it.")
    payload = joblib.load(index_path)
    return payload["model"], payload["filenames"], payload["feature_matrix"]


def find_similar_by_index(
    model: NearestNeighbors,
    filenames: List[str],
    feature_matrix: np.ndarray,
    query_row: int,
    k: int,
) -> List[Tuple[str, float]]:
    """Find the k riffs most similar to an existing library entry.

    Args:
        model: Fitted NearestNeighbors index.
        filenames: Filenames in the same row order the model was
            fitted on.
        feature_matrix: The feature matrix the model was fitted on.
        query_row: Row index (into `filenames` / `feature_matrix`) of
            the riff to find neighbors for.
        k: Number of similar riffs to return, excluding the query
            itself.

    Returns:
        List of (filename, similarity_score) tuples, most similar
        first. `similarity_score` sits in [-1, 1] -- 1 means identical
        direction in feature space, and higher always means more
        similar.
    """
    query_vector = feature_matrix[query_row : query_row + 1]
    return _query_index(model, filenames, query_vector, k, exclude_row=query_row)


def find_similar_by_vector(
    model: NearestNeighbors,
    filenames: List[str],
    query_vector: np.ndarray,
    k: int,
) -> List[Tuple[str, float]]:
    """Find the k riffs most similar to an arbitrary query vector.

    Use this for audio that isn't already in the library: run it
    through extract_features.py's feature functions, transform the
    result with the *same* fitted StandardScaler saved alongside the
    feature matrix (`scaler.joblib`), then pass the resulting vector
    here.

    Args:
        model: Fitted NearestNeighbors index.
        filenames: Filenames in the same row order the model was
            fitted on.
        query_vector: A single standardized feature vector, shape
            `(n_features,)` or `(1, n_features)`.
        k: Number of similar riffs to return.

    Returns:
        List of (filename, similarity_score) tuples, most similar
        first.
    """
    query_vector = np.atleast_2d(query_vector)
    return _query_index(model, filenames, query_vector, k, exclude_row=None)


def _query_index(
    model: NearestNeighbors,
    filenames: List[str],
    query_vector: np.ndarray,
    k: int,
    exclude_row: Optional[int],
) -> List[Tuple[str, float]]:
    """Shared neighbor lookup used by both public query functions."""
    # Ask for one extra neighbor when excluding the query itself, since
    # it would otherwise occupy one of the k requested slots.
    n_neighbors = k + 1 if exclude_row is not None else k
    n_neighbors = min(n_neighbors, len(filenames))
    distances, indices = model.kneighbors(query_vector, n_neighbors=n_neighbors)

    results: List[Tuple[str, float]] = []
    for distance, idx in zip(distances[0], indices[0]):
        if idx == exclude_row:
            continue
        similarity = 1.0 - float(distance)
        results.append((filenames[idx], similarity))
    return results[:k]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for building the search index."""
    parser = argparse.ArgumentParser(
        description=(
            "Build a cosine-distance NearestNeighbors search index over a "
            "normalized feature matrix and save it to disk."
        )
    )
    parser.add_argument(
        "--feature-csv", type=Path, default=DEFAULT_FEATURE_CSV,
        help=f"Path to the normalized feature matrix CSV (default: {DEFAULT_FEATURE_CSV}).",
    )
    parser.add_argument(
        "--index-path", type=Path, default=DEFAULT_INDEX_PATH,
        help=f"Destination path for the saved search index (default: {DEFAULT_INDEX_PATH}).",
    )
    parser.add_argument(
        "--n-neighbors", type=int, default=DEFAULT_N_NEIGHBORS,
        help=f"Default number of neighbors the index is built to return (default: {DEFAULT_N_NEIGHBORS}).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: build and persist the similarity search index."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    filenames, feature_matrix = load_feature_matrix(args.feature_csv)
    model = build_search_index(feature_matrix, args.n_neighbors)
    save_search_index(model, filenames, feature_matrix, args.index_path)


if __name__ == "__main__":
    main()

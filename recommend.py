"""CLI recommendation tool.

Given either the filename of a riff already indexed in the library, or
the path to a brand-new audio file, prints the riffs most similar to
it. New audio is run through the exact same extraction pipeline used
to build the library, then transformed with the exact same fitted
scaler, so it lands in the same feature space before being compared.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd

from extract_features import DEFAULT_N_MFCC, DEFAULT_SAMPLE_RATE, extract_features_from_file
from search_engine import (
    DEFAULT_INDEX_PATH,
    find_similar_by_index,
    find_similar_by_vector,
    load_search_index,
)

logger = logging.getLogger(__name__)

DEFAULT_SCALER_PATH = Path("data/features/scaler.joblib")
DEFAULT_TOP_K = 5


def recommend_from_library(
    filename: str,
    model,
    filenames: List[str],
    feature_matrix: np.ndarray,
    k: int,
) -> List[Tuple[str, float]]:
    """Recommend riffs similar to one already indexed in the library.

    Args:
        filename: The identifier exactly as it appears in the
            `filename` column of feature_matrix.csv (e.g. `riff_a.wav`
            or `subfolder/riff_b.wav`).
        model: Fitted NearestNeighbors index.
        filenames: Filenames in the same row order the model was
            fitted on.
        feature_matrix: The feature matrix the model was fitted on.
        k: Number of recommendations to return.

    Returns:
        List of (filename, similarity_score) tuples, most similar
        first.

    Raises:
        ValueError: If `filename` isn't in the indexed library.
    """
    try:
        query_row = filenames.index(filename)
    except ValueError as exc:
        raise ValueError(
            f"'{filename}' isn't in the indexed library. Known filenames: {filenames}"
        ) from exc
    return find_similar_by_index(model, filenames, feature_matrix, query_row, k)


def recommend_from_new_audio(
    audio_path: Path,
    scaler_path: Path,
    model,
    filenames: List[str],
    k: int,
) -> List[Tuple[str, float]]:
    """Recommend riffs similar to a brand-new audio file, not yet indexed.

    Runs the same extraction pipeline used to build the library
    (`extract_features_from_file`, with the same sample rate and MFCC
    count extract_features.py defaults to) and then applies the exact
    fitted `StandardScaler` the library was normalized with -- never a
    scaler re-fit on this one sample, which would be statistically
    meaningless -- before searching the index.

    Args:
        audio_path: Path to the new audio file.
        scaler_path: Path to the `scaler.joblib` saved by
            extract_features.py.
        model: Fitted NearestNeighbors index.
        filenames: Filenames in the same row order the model was
            fitted on.
        k: Number of recommendations to return.

    Returns:
        List of (filename, similarity_score) tuples, most similar
        first.

    Raises:
        FileNotFoundError: If `audio_path` or `scaler_path` don't
            exist.
        RuntimeError: If feature extraction fails on `audio_path`.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"{audio_path} does not exist.")
    if not scaler_path.exists():
        raise FileNotFoundError(f"{scaler_path} not found -- run extract_features.py first.")

    features = extract_features_from_file(audio_path, DEFAULT_SAMPLE_RATE, DEFAULT_N_MFCC)
    if features is None:
        raise RuntimeError(f"Could not extract features from {audio_path}.")

    scaler = joblib.load(scaler_path)
    # Explicitly align to the column order the scaler was fitted with
    # (scikit-learn records this as feature_names_in_ when fit on a
    # DataFrame, which extract_features.py always fits it on) rather
    # than assuming dict insertion order happens to match -- this also
    # avoids scikit-learn's feature-name mismatch warning on transform.
    expected_columns = list(getattr(scaler, "feature_names_in_", features.keys()))
    raw_vector = pd.DataFrame([features])[expected_columns]
    standardized_vector = scaler.transform(raw_vector)[0]

    return find_similar_by_vector(model, filenames, standardized_vector, k)


def print_recommendations(query_label: str, results: List[Tuple[str, float]]) -> None:
    """Print a formatted list of recommendations to the terminal.

    Args:
        query_label: Human-readable identifier for the query (a
            filename or a path).
        results: List of (filename, similarity_score) tuples to
            display, in order.
    """
    print(f"Riffs most similar to '{query_label}':")
    if not results:
        print("  (no other riffs in the library to compare against)")
        return
    for rank, (filename, similarity) in enumerate(results, start=1):
        print(f"  {rank}. {filename}  (similarity: {similarity:.3f})")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the recommendation CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Find the riffs most similar to a query, either an existing "
            "library entry or a brand-new audio file."
        )
    )
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument(
        "--filename", type=str,
        help="Filename of a riff already in the library, exactly as it "
             "appears in feature_matrix.csv (e.g. 'riff_a.wav').",
    )
    query_group.add_argument(
        "--audio-file", type=Path,
        help="Path to a new audio file not yet in the library.",
    )
    parser.add_argument(
        "--index-path", type=Path, default=DEFAULT_INDEX_PATH,
        help=f"Path to the saved search index (default: {DEFAULT_INDEX_PATH}).",
    )
    parser.add_argument(
        "--scaler-path", type=Path, default=DEFAULT_SCALER_PATH,
        help=f"Path to the fitted scaler, only used with --audio-file (default: {DEFAULT_SCALER_PATH}).",
    )
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K,
        help=f"Number of recommendations to return (default: {DEFAULT_TOP_K}).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: load the index and print recommendations for one query."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    model, filenames, feature_matrix = load_search_index(args.index_path)

    if args.filename is not None:
        results = recommend_from_library(args.filename, model, filenames, feature_matrix, args.top_k)
        query_label = args.filename
    else:
        results = recommend_from_new_audio(args.audio_file, args.scaler_path, model, filenames, args.top_k)
        query_label = str(args.audio_file)

    print_recommendations(query_label, results)


if __name__ == "__main__":
    main()

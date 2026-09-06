"""Audio ingestion and feature extraction engine.

This module scans a directory of audio recordings -- guitar riffs, in
this project's case -- and converts each one into a fixed-length
numeric feature vector suitable for vector similarity search. Four
complementary families of features are extracted per file:

* Chroma STFT       -- pitch-class energy, i.e. *which notes* are played.
* MFCCs             -- spectral envelope, i.e. *what it sounds like*
                        (timbre, tone, "body").
* Spectral contrast -- peak/valley texture across frequency bands,
                        i.e. how bright or harmonically dense it is.
* Tempo (BPM)       -- rhythmic pace of the performance.

Every file in a library is processed into a single table, standardized
to zero mean / unit variance per column, and written out as a CSV
feature matrix alongside the fitted scaler, so the exact same
normalization can be reapplied to new query audio later.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import joblib
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Pitch classes in the order librosa's chroma bins are indexed (starting at C).
PITCH_CLASSES: List[str] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
]

DEFAULT_INPUT_DIR = Path("data/audio")
DEFAULT_OUTPUT_CSV = Path("data/features/feature_matrix.csv")
DEFAULT_SCALER_PATH = Path("data/features/scaler.joblib")
DEFAULT_SAMPLE_RATE = 22050
DEFAULT_N_MFCC = 13
SUPPORTED_EXTENSIONS: Set[str] = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def load_audio(file_path: Path, sample_rate: int) -> Tuple[np.ndarray, int]:
    """Load an audio file, downmixed to mono with silence trimmed.

    Every file is resampled to the same `sample_rate` so extracted
    features stay directly comparable across the library -- without
    this, two identical riffs recorded at different sample rates could
    look artificially dissimilar. Leading/trailing silence is trimmed
    so dead air doesn't dilute the aggregated statistics below.

    Args:
        file_path: Path to the audio file on disk.
        sample_rate: Target sample rate to resample to.

    Returns:
        Tuple of (audio time series, sample rate actually used).
    """
    y, sr = librosa.load(file_path, sr=sample_rate, mono=True)
    y, _ = librosa.effects.trim(y)
    return y, sr


def extract_chroma_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    """Extract chromagram statistics capturing pitch-class (harmonic) content.

    A chroma vector folds all spectral energy into 12 bins, one per
    semitone of the Western scale (C, C#, D, ... B), independent of
    octave. For a guitar riff this captures *which notes and chord
    shapes are being played* -- two riffs built around the same power
    chord or scale pattern land close together in chroma space even if
    they're recorded on different guitars or at different tempos.

    Args:
        y: Audio time series.
        sr: Sample rate of `y`.

    Returns:
        Dictionary of `chroma_mean_<pitch>` / `chroma_std_<pitch>` ->
        value, aggregated over the whole clip.
    """
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    means = np.mean(chroma, axis=1)
    stds = np.std(chroma, axis=1)
    features: Dict[str, float] = {}
    for pitch_class, mean_val, std_val in zip(PITCH_CLASSES, means, stds):
        features[f"chroma_mean_{pitch_class}"] = float(mean_val)
        features[f"chroma_std_{pitch_class}"] = float(std_val)
    return features


def extract_mfcc_features(y: np.ndarray, sr: int, n_mfcc: int) -> Dict[str, float]:
    """Extract MFCC statistics capturing timbre and tonal "body".

    Mel-Frequency Cepstral Coefficients summarize the shape of the
    spectral envelope on a scale approximating human pitch perception.
    Where chroma answers "which notes," MFCCs answer "what does it
    sound like" -- they're sensitive to pickup character, amp voicing,
    distortion, and playing dynamics, which is what lets the
    similarity engine separate a bright, clean riff from a warm,
    heavily palm-muted one even when the underlying notes match.

    Args:
        y: Audio time series.
        sr: Sample rate of `y`.
        n_mfcc: Number of MFCC coefficients to compute.

    Returns:
        Dictionary of `mfcc_mean_<n>` / `mfcc_std_<n>` -> value,
        aggregated over the whole clip.
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    means = np.mean(mfcc, axis=1)
    stds = np.std(mfcc, axis=1)
    features: Dict[str, float] = {}
    for idx, (mean_val, std_val) in enumerate(zip(means, stds), start=1):
        features[f"mfcc_mean_{idx}"] = float(mean_val)
        features[f"mfcc_std_{idx}"] = float(std_val)
    return features


def extract_spectral_contrast_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    """Extract spectral contrast statistics capturing tonal texture.

    Spectral contrast measures the level gap between spectral peaks
    and valleys within several frequency sub-bands. A riff with clear
    harmonic overtones and string separation -- an open, ringing
    arpeggio -- produces high contrast; a dense, heavily distorted or
    muted riff produces low contrast because the spectrum fills in
    more uniformly. This gives the engine a sense of brightness and
    texture that's independent of both pitch and timbre.

    Args:
        y: Audio time series.
        sr: Sample rate of `y`.

    Returns:
        Dictionary of `spectral_contrast_mean_band_<n>` /
        `spectral_contrast_std_band_<n>` -> value, aggregated over the
        whole clip.
    """
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    means = np.mean(contrast, axis=1)
    stds = np.std(contrast, axis=1)
    features: Dict[str, float] = {}
    for idx, (mean_val, std_val) in enumerate(zip(means, stds), start=1):
        features[f"spectral_contrast_mean_band_{idx}"] = float(mean_val)
        features[f"spectral_contrast_std_band_{idx}"] = float(std_val)
    return features


def extract_tempo_feature(y: np.ndarray, sr: int) -> Dict[str, float]:
    """Extract the estimated tempo, capturing the riff's rhythmic pace.

    Tempo, in beats per minute, is a coarse but musically meaningful
    signal on its own: a blistering 220 BPM shred pattern and a
    laid-back 80 BPM groove are rhythmically dissimilar even when they
    share harmonic content.

    Args:
        y: Audio time series.
        sr: Sample rate of `y`.

    Returns:
        Dictionary with a single key, `tempo_bpm`.
    """
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    # librosa's tempo return type has varied across versions (a plain
    # float in some, a length-1 ndarray in others, to support
    # multi-channel input) -- normalize defensively so downstream code
    # never has to care which one it got.
    tempo_value = float(np.atleast_1d(tempo)[0])
    return {"tempo_bpm": tempo_value}


def extract_features_from_file(
    file_path: Path,
    sample_rate: int,
    n_mfcc: int,
) -> Optional[Dict[str, float]]:
    """Run the full extraction pipeline on a single audio file.

    Args:
        file_path: Path to the audio file.
        sample_rate: Target sample rate for loading.
        n_mfcc: Number of MFCC coefficients to compute.

    Returns:
        Flat dictionary of every extracted feature, or None if the
        file could not be processed (corrupt, unreadable, or silent).
    """
    try:
        y, sr = load_audio(file_path, sample_rate)
        if y.size == 0:
            logger.warning("Skipping %s: file is empty or silent after trimming.", file_path)
            return None
        features: Dict[str, float] = {}
        features.update(extract_chroma_features(y, sr))
        features.update(extract_mfcc_features(y, sr, n_mfcc))
        features.update(extract_spectral_contrast_features(y, sr))
        features.update(extract_tempo_feature(y, sr))
        return features
    except Exception:
        logger.exception("Failed to extract features from %s -- skipping.", file_path)
        return None


def discover_audio_files(input_dir: Path, extensions: Set[str]) -> List[Path]:
    """Recursively find audio files under `input_dir` with a supported extension.

    Args:
        input_dir: Root directory to search.
        extensions: Lowercase extensions (with leading dot) to include.

    Returns:
        Sorted list of matching file paths.
    """
    return sorted(
        p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in extensions
    )


def build_feature_matrix(
    input_dir: Path,
    sample_rate: int,
    n_mfcc: int,
    audio_extensions: Set[str],
) -> pd.DataFrame:
    """Extract features for every audio file in `input_dir` into one table.

    Args:
        input_dir: Root directory containing the audio library, searched
            recursively so riffs can be organized into subfolders.
        sample_rate: Target sample rate for loading each file.
        n_mfcc: Number of MFCC coefficients to compute per file.
        audio_extensions: Lowercase extensions (with leading dot)
            treated as audio, e.g. {".wav", ".mp3"}.

    Returns:
        DataFrame with one row per successfully processed file. The
        first column, `filename`, identifies each track by its path
        relative to `input_dir`; every other column is a raw
        (not yet normalized) numeric feature.

    Raises:
        FileNotFoundError: If no supported audio files exist under
            `input_dir`.
        RuntimeError: If every discovered file failed to process.
    """
    audio_files = discover_audio_files(input_dir, audio_extensions)
    if not audio_files:
        raise FileNotFoundError(
            f"No audio files with extensions {sorted(audio_extensions)} found under {input_dir}"
        )

    rows: List[Dict[str, Any]] = []
    for file_path in tqdm(audio_files, desc="Extracting features", unit="file"):
        features = extract_features_from_file(file_path, sample_rate, n_mfcc)
        if features is None:
            continue
        row: Dict[str, Any] = {"filename": str(file_path.relative_to(input_dir))}
        row.update(features)
        rows.append(row)

    if not rows:
        raise RuntimeError("All discovered audio files failed to process; see warnings above.")

    logger.info("Successfully extracted features for %d/%d files.", len(rows), len(audio_files))
    return pd.DataFrame(rows)


def normalize_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """Standardize every numeric feature column to zero mean, unit variance.

    Cosine similarity, used downstream by the search engine, is
    invariant to a vector's overall magnitude but not to per-feature
    scale. Raw MFCC coefficients can range over tens of units while
    chroma values sit between 0 and 1 and tempo sits in the
    tens-to-hundreds of BPM -- left unscaled, MFCCs alone would
    dominate every similarity score. Standardizing each column
    independently puts pitch, timbre, texture, and tempo on equal
    footing before comparison.

    Args:
        df: Feature matrix with a `filename` column plus numeric
            feature columns.

    Returns:
        Tuple of (new DataFrame with numeric columns replaced by their
        standardized values, the fitted StandardScaler so the exact
        same transformation can be reapplied to new query audio later).
    """
    feature_columns = [c for c in df.columns if c != "filename"]
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df[feature_columns])
    scaled_df = df.copy()
    scaled_df[feature_columns] = scaled_values
    return scaled_df, scaler


def save_outputs(
    df: pd.DataFrame,
    scaler: StandardScaler,
    output_csv: Path,
    scaler_path: Path,
) -> None:
    """Persist the normalized feature matrix and fitted scaler to disk.

    The scaler is saved rather than discarded so that any new query
    audio processed later -- by the search engine or the
    recommendation CLI -- can be transformed with the exact mean and
    variance the library was normalized with, instead of being
    incorrectly re-fit on a single sample.

    Args:
        df: Normalized feature matrix to write as CSV.
        scaler: Fitted StandardScaler to persist.
        output_csv: Destination path for the CSV file.
        scaler_path: Destination path for the serialized scaler.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    joblib.dump(scaler, scaler_path)
    logger.info("Feature matrix written to %s", output_csv)
    logger.info("Fitted scaler written to %s", scaler_path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the feature extraction pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract chroma, MFCC, spectral contrast, and tempo features "
            "from a library of guitar riff audio files into a single "
            "normalized CSV feature matrix."
        )
    )
    parser.add_argument(
        "--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
        help=f"Directory containing the audio library, searched recursively (default: {DEFAULT_INPUT_DIR}).",
    )
    parser.add_argument(
        "--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV,
        help=f"Destination path for the normalized feature matrix CSV (default: {DEFAULT_OUTPUT_CSV}).",
    )
    parser.add_argument(
        "--scaler-path", type=Path, default=DEFAULT_SCALER_PATH,
        help=f"Destination path for the fitted scaler artifact (default: {DEFAULT_SCALER_PATH}).",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE,
        help=f"Target sample rate all audio is resampled to (default: {DEFAULT_SAMPLE_RATE}).",
    )
    parser.add_argument(
        "--n-mfcc", type=int, default=DEFAULT_N_MFCC,
        help=f"Number of MFCC coefficients to compute per file (default: {DEFAULT_N_MFCC}).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: build, normalize, and persist the feature matrix."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    df = build_feature_matrix(args.input_dir, args.sample_rate, args.n_mfcc, SUPPORTED_EXTENSIONS)
    normalized_df, scaler = normalize_feature_matrix(df)
    save_outputs(normalized_df, scaler, args.output_csv, args.scaler_path)


if __name__ == "__main__":
    main()

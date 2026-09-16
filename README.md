# 🎸 Acoustic Guitar Riff Similarity Finder

![Python](https://img.shields.io/badge/python-3.12-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-FF4B4B)
![scikit-learn](https://img.shields.io/badge/built%20with-scikit--learn-F7931E)

Content-based similarity search for guitar riffs. Every riff in a library is reduced to a fixed-length audio-feature vector — pitch content, timbre, spectral texture, and tempo — and compared against every other riff using cosine similarity, so "find something that sounds like this" becomes a nearest-neighbor lookup rather than a metadata or tag match.

No deep learning involved: the whole pipeline is classical digital signal processing (via [librosa](https://librosa.org/)) feeding a classical nearest-neighbors model (via [scikit-learn](https://scikit-learn.org/)).

## How it works

```
                    ┌────────────────────────┐
   audio library →  │  extract_features.py    │  →  feature_matrix.csv
   (.wav/.mp3/...)   │  (chroma, MFCC,         │     scaler.joblib
                    │   spectral contrast,     │
                    │   tempo → standardized)  │
                    └────────────────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │    search_engine.py     │  →  search_index.joblib
                    │ (cosine NearestNeighbors)│
                    └────────────────────────┘
                                │
                 ┌──────────────┴───────────────┐
                 ▼                               ▼
        ┌─────────────────┐             ┌─────────────────┐
        │   recommend.py   │             │      app.py      │
        │  CLI lookup tool  │             │  Streamlit UI    │
        └─────────────────┘             └─────────────────┘
```

Each stage is a standalone module that reads/writes plain files (CSV, joblib artifacts), so any stage can be rerun independently, imported elsewhere, or swapped out without touching the others.

### What's actually being measured

Four feature families are extracted per riff, aggregated (mean and standard deviation) across the whole clip:

| Feature | Captures | Why it matters for a riff |
|---|---|---|
| **Chroma STFT** | Pitch-class energy across the 12 semitones | *Which notes and chord shapes* are being played — the harmonic fingerprint |
| **MFCCs** | Spectral envelope, on a perceptual scale | *What it sounds like* — tone, pickup character, distortion, "body" |
| **Spectral contrast** | Peak-to-valley energy across frequency bands | Brightness and harmonic density — a ringing arpeggio vs. a dense, muted riff |
| **Tempo (BPM)** | Rhythmic pace | A 220 BPM shred pattern and an 80 BPM groove are rhythmically distinct, whatever notes they share |

That's 65 numeric features per riff. Before comparison, every column is standardized to zero mean / unit variance — cosine similarity is scale-invariant *per vector*, but not across features with wildly different native ranges (raw MFCC magnitude would otherwise dominate a 0–1-ranged chroma value in every comparison). The fitted scaler is saved alongside the feature matrix so any later query — a new upload, a CLI lookup — gets transformed with the exact statistics the library was built on, never re-fit on a single sample.

## Project structure

```
acoustic-guitar-riff-similarity-finder/
├── extract_features.py     # audio ingestion + feature extraction
├── search_engine.py        # cosine-distance NearestNeighbors index
├── recommend.py             # CLI: query by filename or new audio
├── app.py                   # Streamlit UI
├── requirements.txt
├── .gitignore
├── LICENSE
└── data/
    ├── audio/                # riff library (not versioned by default)
    └── features/             # generated: feature_matrix.csv, scaler.joblib, search_index.joblib
```

## Installation

```bash
git clone https://github.com/SKYLORD69-PY/acoustic-guitar-riff-similarity-finder.git
cd acoustic-guitar-riff-similarity-finder

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Verified against Python 3.12 and the exact dependency versions pinned in `requirements.txt`.

## Getting audio samples

`data/audio/` is empty by default — riff audio is left out of version control (see [Licensing note](#licensing-note) below). To populate it:

- **Your own recordings** — a handful of short clips is enough to see the pipeline work.
- **Free, royalty-free loops** — [Looperman](https://www.looperman.com/loops/tags/free-guitar-riff-loops-samples-sounds-wavs-download), [Sample Focus](https://samplefocus.com/categories/guitar).
- **Academic datasets** — [GuitarSet](https://github.com/marl/GuitarSet) (NYU, 360 annotated recordings), [IDMT-SMT-GUITAR](https://www.idmt.fraunhofer.de/en/publications/datasets/guitar.html) (Fraunhofer, 7 guitars/multiple techniques — CC BY-NC-ND, non-commercial only).

Any mix of `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a` works; subfolders are fine, the extractor scans recursively.

## Usage

**1. Extract and normalize features**

```bash
python extract_features.py \
  --input-dir data/audio \
  --output-csv data/features/feature_matrix.csv \
  --scaler-path data/features/scaler.joblib
```

**2. Build the similarity index**

```bash
python search_engine.py \
  --feature-csv data/features/feature_matrix.csv \
  --index-path data/features/search_index.joblib
```

**3. Get recommendations from the command line**

```bash
# against a riff already in the library
python recommend.py --filename riff_a.wav --top-k 5

# against a brand-new file, not yet in the library
python recommend.py --audio-file path/to/new_riff.wav --top-k 5
```

**4. Launch the interactive UI**

```bash
streamlit run app.py
```

The UI lets you pick a library riff or upload a new one, then shows the top matches with audio playback, an interactive waveform comparison against the best match, and a similarity-distribution chart across the whole library. Cyan consistently marks the query signal and amber consistently marks matches against it, across every chart and panel.

## Tech stack

- **[librosa](https://librosa.org/)** — audio loading and feature extraction
- **[scikit-learn](https://scikit-learn.org/)** — `StandardScaler`, `NearestNeighbors` (cosine distance)
- **[pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/)** — feature matrix handling
- **[Streamlit](https://streamlit.io/)** — interactive UI
- **[Plotly](https://plotly.com/python/)** — waveform and distribution charts

## Possible extensions

- Approximate nearest-neighbor indexing (e.g. FAISS, Annoy) for libraries too large for brute-force cosine search
- A playing-technique or genre classifier layered on top of the existing features
- Pitch- and tempo-normalized features, for matching the same riff played in a different key or speed
- A hosted public demo of the Streamlit app

## Licensing note

The code in this repository is MIT licensed (see `LICENSE`). Audio you add to `data/audio/` is not covered by that license — if you populate it from a dataset with its own terms (e.g. IDMT-SMT-GUITAR's non-commercial clause), those terms still apply to the audio itself.

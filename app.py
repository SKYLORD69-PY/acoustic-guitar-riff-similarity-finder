"""Interactive Streamlit UI for exploring riff similarity.

Styled as a signal-analysis instrument rather than a generic media app:
riffs are treated as signals to be measured, not tracks to be browsed.
Cyan consistently marks the query signal; amber consistently marks
matches against it -- in the waveform view, the per-match meters, and
the library-wide distribution chart -- so color carries a fixed
meaning throughout instead of decorating at random.

Lets a user pick a riff already in the indexed library, or upload a
brand-new one, and see its closest matches: an interactive waveform
comparison, audio playback, and where those matches sit in the full
distribution of similarity scores against the rest of the library --
not just the top-k winners in isolation.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from recommend import DEFAULT_SCALER_PATH, recommend_from_new_audio
from search_engine import DEFAULT_INDEX_PATH, find_similar_by_index, load_search_index

DEFAULT_AUDIO_ROOT = Path("data/audio")
DEFAULT_UPLOAD_SCRATCH_DIR = Path("data/uploads")
DEFAULT_TOP_K = 5

# Fixed semantic palette: cyan is always the query signal, amber is
# always a match against it. Reused identically across the CSS, the
# match meters, and both Plotly figures so the color coding stays
# legible without a legend.
COLOR_BG_VOID = "#0A0E14"
COLOR_BG_PANEL = "#12181F"
COLOR_QUERY = "#38BDF8"
COLOR_QUERY_FILL = "rgba(56, 189, 248, 0.20)"
COLOR_MATCH = "#F59E0B"
COLOR_MATCH_FILL = "rgba(245, 158, 11, 0.20)"
COLOR_TEXT_MUTED = "#6B7686"
COLOR_METER_BASE = (0x3A, 0x42, 0x50)
COLOR_METER_PEAK = (0xF5, 0x9E, 0x0B)


def inject_custom_css() -> None:
    """Apply the app's visual identity: fonts, palette, and panel styling.

    Everything here is cosmetic -- no functional behavior depends on
    it -- so a browser that ignores the injected `<style>` block still
    gets a fully working app, just with Streamlit's defaults instead.
    """
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;700&display=swap');

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; }

        .rf-hero-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2.05rem; font-weight: 700;
            color: #E5E9F0; margin-bottom: 0.2rem;
        }
        .rf-hero-sub { color: #6B7686; font-size: 0.95rem; margin-bottom: 1rem; }

        .rf-stat-strip {
            display: flex; gap: 2rem;
            padding: 0.6rem 0 1.2rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 1.3rem;
        }
        .rf-stat { display: flex; flex-direction: column; }
        .rf-stat-value {
            font-family: 'JetBrains Mono', monospace;
            color: #38BDF8; font-size: 1.1rem; font-weight: 700;
        }
        .rf-stat-label { color: #6B7686; font-size: 0.75rem; margin-top: 0.1rem; }

        .rf-query-line {
            font-family: 'Space Grotesk', sans-serif;
            color: #E5E9F0; font-size: 1.05rem;
            margin: 0.2rem 0 0.6rem 0;
        }
        .rf-query-line .rf-accent { color: #38BDF8; }

        .rf-panel {
            background: #12181F;
            border-left: 4px solid rgba(245, 158, 11, 0.30);
            border-radius: 3px;
            padding: 0.7rem 1rem 0.55rem 1rem;
            margin-bottom: 0.3rem;
        }
        .rf-panel-top {
            border-left: 4px solid #F59E0B;
            box-shadow: 0 0 16px rgba(245, 158, 11, 0.12);
        }
        .rf-panel-row {
            display: flex; justify-content: space-between; align-items: baseline;
            margin-bottom: 0.4rem; gap: 0.75rem;
        }
        .rf-panel-name { font-weight: 600; color: #E5E9F0; font-size: 0.93rem; }
        .rf-panel-rank {
            font-family: 'JetBrains Mono', monospace; color: #6B7686;
            font-size: 0.78rem; margin-right: 0.5rem;
        }
        .rf-readout {
            font-family: 'JetBrains Mono', monospace; font-weight: 700;
            color: #F59E0B; font-size: 0.98rem; white-space: nowrap;
        }
        .rf-meter-track {
            position: relative; height: 6px;
            background: rgba(255,255,255,0.06); border-radius: 3px; overflow: hidden;
        }
        .rf-meter-fill {
            position: absolute; top: 0; left: 0; height: 100%; border-radius: 3px;
        }
        .rf-meter-zero { position: absolute; top: -2px; bottom: -2px; left: 50%; width: 1px; background: rgba(255,255,255,0.22); }

        @keyframes rf-fade-up { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
        .rf-results-block { animation: rf-fade-up 0.45s ease-out; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_search_index(index_path: Path):
    """Load and cache the search index artifact for the Streamlit session.

    Streamlit reruns the whole script on every widget interaction, so
    the index is deserialized once per session and cached rather than
    reloaded from disk on every click.

    Args:
        index_path: Path to the artifact written by search_engine.py.

    Returns:
        Tuple of (fitted NearestNeighbors index, filenames, feature
        matrix), as returned by `load_search_index`.
    """
    return load_search_index(index_path)


def resolve_audio_path(filename: str, audio_root: Path) -> Path:
    """Resolve a library `filename` (as stored in the CSV) to a playable path.

    Args:
        filename: The `filename` value from feature_matrix.csv --
            a path relative to the audio library root.
        audio_root: The directory the library was extracted from.

    Returns:
        Absolute-or-relative path suitable for `librosa.load` / `st.audio`.
    """
    return audio_root / filename


def full_similarity_distribution(
    model,
    filenames: List[str],
    feature_matrix: np.ndarray,
    query_row: int,
) -> List[Tuple[str, float]]:
    """Similarity of one library entry against every other riff, unfiltered.

    Distinct from `find_similar_by_index`, which is meant to return
    only the top k: this returns every comparison, so the UI can plot
    where the library's full similarity distribution sits rather than
    only ever showing the winners.

    Args:
        model: Fitted NearestNeighbors index.
        filenames: Filenames in the same row order the model was
            fitted on.
        feature_matrix: The feature matrix the model was fitted on.
        query_row: Row index of the riff to compare against everything
            else.

    Returns:
        List of (filename, similarity_score) tuples for every other
        riff in the library, most similar first.
    """
    n_other_riffs = len(filenames) - 1
    return find_similar_by_index(model, filenames, feature_matrix, query_row, k=n_other_riffs)


def save_uploaded_file(uploaded_file, destination_dir: Path) -> Path:
    """Persist an in-memory Streamlit upload to a real path on disk.

    Feature extraction reads from a filesystem path, not an in-memory
    buffer, so the upload is written out once, to a scratch directory
    that survives across Streamlit reruns (unlike a context-managed
    temp directory, which would be deleted before the audio is played
    back later in the same run).

    Args:
        uploaded_file: The object returned by `st.file_uploader`.
        destination_dir: Directory to write the file into.

    Returns:
        Path to the saved file.
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination_dir / uploaded_file.name
    destination_path.write_bytes(uploaded_file.getvalue())
    return destination_path


def similarity_to_meter_color(score: float) -> str:
    """Map a similarity score to a point on the muted-slate-to-amber scale.

    Only the positive half of the [-1, 1] range ramps toward amber --
    a riff has to actually resemble the query to register as "hot" on
    the meter; anything at or below zero reads as the same muted base
    color regardless of how negative it gets.

    Args:
        score: Cosine similarity in [-1, 1].

    Returns:
        A `rgb(r, g, b)` CSS color string.
    """
    t = max(0.0, min(1.0, score))
    rgb = tuple(int(COLOR_METER_BASE[i] + (COLOR_METER_PEAK[i] - COLOR_METER_BASE[i]) * t) for i in range(3))
    return f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"


def render_match_panel_html(rank: int, filename: str, similarity: float, is_top: bool) -> str:
    """Build the HTML for one match's rank, meter, and numeric readout.

    Args:
        rank: 1-based rank of this match among the returned results.
        filename: Filename to display.
        similarity: Cosine similarity score in [-1, 1].
        is_top: Whether this is the single best match, which gets a
            visually distinct treatment rather than a text badge --
            the glow itself is the signal.

    Returns:
        A self-contained HTML string ready for `st.markdown` with
        `unsafe_allow_html=True`.
    """
    pct = max(0.0, min(100.0, (similarity + 1) / 2 * 100))
    color = similarity_to_meter_color(similarity)
    panel_class = "rf-panel rf-panel-top" if is_top else "rf-panel"
    safe_name = html.escape(filename)
    return f"""
    <div class="{panel_class}">
      <div class="rf-panel-row">
        <span><span class="rf-panel-rank">{rank:02d}</span><span class="rf-panel-name">{safe_name}</span></span>
        <span class="rf-readout">{similarity:.3f}</span>
      </div>
      <div class="rf-meter-track">
        <div class="rf-meter-zero"></div>
        <div class="rf-meter-fill" style="width:{pct:.1f}%; background:{color};"></div>
      </div>
    </div>
    """


def downsample_waveform_envelope(y: np.ndarray, target_points: int = 600) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce an audio signal to a min/max envelope for fast, clean plotting.

    Plotting every raw sample of even a few seconds of audio -- tens
    of thousands of points -- is slow to render in a browser and looks
    like noise rather than a waveform. Chunking the signal and keeping
    the min and max of each chunk reproduces the waveform's outline;
    it's the same technique real audio editors use for their waveform
    views.

    Args:
        y: Audio time series.
        target_points: Approximate number of chunks to reduce to.

    Returns:
        Tuple of (per-chunk minimum, per-chunk maximum) arrays.
    """
    if len(y) <= target_points:
        return y, y
    chunk_size = max(1, len(y) // target_points)
    n_chunks = len(y) // chunk_size
    trimmed = y[: n_chunks * chunk_size].reshape(n_chunks, chunk_size)
    return trimmed.min(axis=1), trimmed.max(axis=1)


def build_waveform_figure(
    query_label: str,
    query_y: np.ndarray,
    query_sr: int,
    match_label: str,
    match_y: np.ndarray,
    match_sr: int,
) -> go.Figure:
    """Build a stacked, interactive waveform comparison figure.

    Args:
        query_label: Display name for the query riff.
        query_y: Query audio time series.
        query_sr: Query sample rate.
        match_label: Display name for the matched riff.
        match_y: Matched audio time series.
        match_sr: Matched sample rate.

    Returns:
        Plotly figure with the query on top (cyan) and the best match
        below (amber), each independently scaled on the time axis
        since the two clips may run different lengths.
    """
    fig = make_subplots(rows=2, cols=1, subplot_titles=(f"Query -- {query_label}", f"Best match -- {match_label}"))

    for row, (y, sr, color, fill_color) in enumerate(
        [(query_y, query_sr, COLOR_QUERY, COLOR_QUERY_FILL), (match_y, match_sr, COLOR_MATCH, COLOR_MATCH_FILL)],
        start=1,
    ):
        lo, hi = downsample_waveform_envelope(y)
        t = np.linspace(0, len(y) / sr, len(hi))
        fig.add_trace(
            go.Scatter(x=t, y=hi, mode="lines", line=dict(color=color, width=1), showlegend=False, hoverinfo="skip"),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=lo,
                mode="lines",
                line=dict(color=color, width=1),
                fill="tonexty",
                fillcolor=fill_color,
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=1,
        )

    fig.update_layout(
        height=420,
        paper_bgcolor=COLOR_BG_VOID,
        plot_bgcolor=COLOR_BG_PANEL,
        font=dict(color=COLOR_TEXT_MUTED, family="Inter, sans-serif"),
        margin=dict(l=45, r=20, t=45, b=35),
        showlegend=False,
    )
    fig.update_xaxes(title_text="seconds", showgrid=False, zeroline=False, color=COLOR_TEXT_MUTED)
    fig.update_yaxes(showgrid=False, zeroline=True, zerolinecolor="rgba(255,255,255,0.12)", color=COLOR_TEXT_MUTED)
    return fig


def build_distribution_figure(results: List[Tuple[str, float]], top_k: int) -> go.Figure:
    """Build an interactive bar chart of similarity across the whole library.

    Every bar is colored by its own score on the same muted-to-amber
    scale used on the per-match meters, and the bars making up the
    returned top-k get an amber outline -- so the chart shows both the
    overall shape of the distribution and exactly where the cutoff for
    "recommended" fell within it.

    Args:
        results: (filename, similarity_score) tuples, most similar
            first, covering the whole library.
        top_k: Number of leading entries to outline as the returned
            recommendations.

    Returns:
        Plotly bar chart with one bar per comparison.
    """
    names = [name for name, _ in results]
    scores = [score for _, score in results]
    bar_colors = [similarity_to_meter_color(s) for s in scores]
    outline_colors = [COLOR_MATCH if i < top_k else "rgba(0,0,0,0)" for i in range(len(scores))]

    fig = go.Figure(
        go.Bar(
            x=list(range(len(scores))),
            y=scores,
            marker=dict(color=bar_colors, line=dict(color=outline_colors, width=1.5)),
            customdata=names,
            hovertemplate="%{customdata}<br>similarity: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.15)", line_width=1)
    fig.update_layout(
        height=340,
        paper_bgcolor=COLOR_BG_VOID,
        plot_bgcolor=COLOR_BG_PANEL,
        font=dict(color=COLOR_TEXT_MUTED, family="Inter, sans-serif"),
        margin=dict(l=45, r=20, t=30, b=35),
        xaxis=dict(title="riffs, ranked by similarity", showticklabels=False, showgrid=False),
        yaxis=dict(title="cosine similarity", showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
    )
    return fig


def main() -> None:
    """Entry point: render the Streamlit app."""
    st.set_page_config(page_title="Acoustic Guitar Riff Similarity Finder", page_icon="🎸", layout="wide")
    inject_custom_css()

    index_path = Path(st.sidebar.text_input("Search index path", str(DEFAULT_INDEX_PATH)))
    audio_root = Path(st.sidebar.text_input("Audio library root", str(DEFAULT_AUDIO_ROOT)))
    top_k = st.sidebar.slider("Number of matches to show", min_value=1, max_value=10, value=DEFAULT_TOP_K)

    try:
        model, filenames, feature_matrix = get_search_index(index_path)
    except FileNotFoundError as exc:
        st.error(f"{exc} Build it with search_engine.py first.")
        st.stop()
        return

    st.markdown(
        f"""
        <div class="rf-hero-title">Acoustic Guitar Riff Similarity Finder</div>
        <div class="rf-hero-sub">Cosine-similarity search over chroma, MFCC, spectral contrast, and tempo features.</div>
        <div class="rf-stat-strip">
          <div class="rf-stat"><span class="rf-stat-value">{len(filenames)}</span><span class="rf-stat-label">riffs indexed</span></div>
          <div class="rf-stat"><span class="rf-stat-value">{feature_matrix.shape[1]}</span><span class="rf-stat-label">features / riff</span></div>
          <div class="rf-stat"><span class="rf-stat-value">cosine</span><span class="rf-stat-label">distance metric</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    query_mode = st.sidebar.radio("Query with", ["A riff already in the library", "A new upload"])

    if query_mode == "A riff already in the library":
        query_filename = st.sidebar.selectbox("Riff", filenames)
        query_row = filenames.index(query_filename)
        results = full_similarity_distribution(model, filenames, feature_matrix, query_row)
        query_audio_path = resolve_audio_path(query_filename, audio_root)
        query_label = query_filename
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload a riff", type=["wav", "mp3", "flac", "ogg", "m4a"]
        )
        if uploaded_file is None:
            st.info("Upload an audio file in the sidebar to get recommendations.")
            st.stop()
            return
        query_audio_path = save_uploaded_file(uploaded_file, DEFAULT_UPLOAD_SCRATCH_DIR)
        results = recommend_from_new_audio(
            query_audio_path, DEFAULT_SCALER_PATH, model, filenames, k=len(filenames)
        )
        query_label = uploaded_file.name

    top_results = results[:top_k]

    st.markdown(
        f'<div class="rf-query-line">Comparing against <span class="rf-accent">{html.escape(query_label)}</span></div>',
        unsafe_allow_html=True,
    )
    if query_audio_path.exists():
        st.audio(str(query_audio_path))

    tab_matches, tab_waveforms, tab_distribution = st.tabs(["Matches", "Waveforms", "Distribution"])

    with tab_matches:
        st.markdown('<div class="rf-results-block">', unsafe_allow_html=True)
        for rank, (filename, similarity) in enumerate(top_results, start=1):
            st.markdown(render_match_panel_html(rank, filename, similarity, is_top=(rank == 1)), unsafe_allow_html=True)
            match_path = resolve_audio_path(filename, audio_root)
            if match_path.exists():
                st.audio(str(match_path))
            else:
                st.caption(f"Audio file not found at {match_path}")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_waveforms:
        if not top_results:
            st.caption("No matches to compare -- the library only has this one riff.")
        else:
            best_match_path = resolve_audio_path(top_results[0][0], audio_root)
            if query_audio_path.exists() and best_match_path.exists():
                query_y, query_sr = librosa.load(query_audio_path, sr=None, mono=True)
                match_y, match_sr = librosa.load(best_match_path, sr=None, mono=True)
                # Normalize to int at the source -- see the matching comment
                # in extract_features.load_audio for why.
                query_sr, match_sr = int(query_sr), int(match_sr)
                fig = build_waveform_figure(query_label, query_y, query_sr, top_results[0][0], match_y, match_sr)
                st.plotly_chart(fig, width='stretch')
            else:
                st.caption("Can't render waveforms -- one of the audio files is missing on disk.")

    with tab_distribution:
        if results:
            st.plotly_chart(build_distribution_figure(results, top_k), width='stretch')
        else:
            st.caption("Nothing to compare against yet.")


if __name__ == "__main__":
    main()

"""Pitch and harmony analysis behind singing scores.

Karaoke tracks carry no melody data, so a singer is judged against the music
itself: what notes the backing track is playing at each moment (its chroma),
and whether the note being sung belongs among them.
"""

import numpy as np

# The backing track is analysed at a low rate: harmony lives below ~2 kHz.
BACKING_SAMPLERATE = 11025
CHROMA_HOP_S = 0.1
# 0.37 s at 11025 Hz: bins 2.7 Hz apart, fine enough to split semitones down to ~100 Hz.
_CHROMA_WINDOW = 4096
_CHROMA_MIN_HZ = 80.0
_CHROMA_MAX_HZ = 2000.0
# Frames analysed at once, bounding memory on a Pi (~8 MB per batch).
_CHROMA_BATCH = 256

# Range of a singing voice, bass to soprano.
MIN_VOICE_HZ = 70.0
MAX_VOICE_HZ = 1100.0
# YIN aperiodicity at or under which a frame counts as a sung note.
_YIN_THRESHOLD = 0.2

# How far, in semitones, a sung note may stray and still count as on the note.
_PITCH_TOLERANCE = 0.4


def backing_chroma(pcm: np.ndarray, samplerate: int = BACKING_SAMPLERATE) -> np.ndarray:
    """Energy of each of the 12 pitch classes, one row per CHROMA_HOP_S of audio.

    Rows are raw energies; harmony_profile() turns one into what a singer is
    compared against.
    """
    hop = int(samplerate * CHROMA_HOP_S)
    if len(pcm) < _CHROMA_WINDOW:
        return np.zeros((0, 12))
    n_frames = 1 + (len(pcm) - _CHROMA_WINDOW) // hop

    freqs = np.fft.rfftfreq(_CHROMA_WINDOW, 1 / samplerate)
    band = (freqs >= _CHROMA_MIN_HZ) & (freqs <= _CHROMA_MAX_HZ)
    pitch_class = np.round(12 * np.log2(freqs[band] / 440) + 69).astype(int) % 12
    # (bins, 12) one-hot, so summing bins into classes is one matrix product.
    to_chroma = np.eye(12)[pitch_class]
    window = np.hanning(_CHROMA_WINDOW)
    frames = np.lib.stride_tricks.sliding_window_view(pcm, _CHROMA_WINDOW)[::hop]

    chroma = np.empty((n_frames, 12))
    for start in range(0, n_frames, _CHROMA_BATCH):
        batch = frames[start : start + _CHROMA_BATCH] * window
        spectrum = np.abs(np.fft.rfft(batch, axis=1))[:, band]
        chroma[start : start + len(batch)] = spectrum @ to_chroma
    return chroma


def harmony_profile(local: np.ndarray, song: np.ndarray) -> np.ndarray:
    """What a sung note is judged against at one moment, scaled 0-1 per class.

    Each class's energy is taken relative to the median class, so drums and
    noise, which light every class alike, count for nothing and only notes
    standing out are rewarded. The song-wide profile (its key) is blended in
    so a melody note passing between chord tones still counts, and so quiet
    or percussive moments fall back on the key.
    """
    profile = 0.7 * _contrast(local) + 0.3 * _contrast(song)
    peak = profile.max()
    return profile / peak if peak > 0 else profile


def _contrast(chroma: np.ndarray) -> np.ndarray:
    standout = np.clip(chroma - np.median(chroma), 0, None)
    peak = standout.max()
    return standout / peak if peak > 0 else standout


def detect_pitch(frame: np.ndarray, samplerate: int) -> float | None:
    """Fundamental frequency of a sung frame in Hz (YIN), or None if unpitched.

    The frame must span at least pitch_frame_length(samplerate) samples.
    """
    max_lag = int(samplerate / MIN_VOICE_HZ)
    min_lag = int(samplerate / MAX_VOICE_HZ)
    x = frame.astype(np.float64)
    width = len(x) - max_lag
    if width < max_lag:
        raise ValueError("frame too short for the lowest voice pitch")

    # Difference function d(tau) = sum (x[j] - x[j+tau])^2 over a fixed width,
    # expanded into energies and an FFT cross-correlation.
    n_fft = 1 << int(np.ceil(np.log2(len(x) + width)))
    corr = np.fft.irfft(np.fft.rfft(x, n_fft) * np.conj(np.fft.rfft(x[:width], n_fft)), n_fft)
    corr = corr[: max_lag + 1]
    energy = np.concatenate(([0.0], np.cumsum(x * x)))
    lags = np.arange(max_lag + 1)
    diff = energy[width] + (energy[lags + width] - energy[lags]) - 2 * corr

    # Cumulative mean normalised difference: dips well under 1 at the period.
    running = np.cumsum(diff[1:])
    cmnd = np.ones(max_lag + 1)
    np.divide(diff[1:] * lags[1:], running, out=cmnd[1:], where=running > 0)

    below = np.nonzero(cmnd[min_lag:] < _YIN_THRESHOLD)[0]
    if len(below) == 0:
        return None
    tau = min_lag + below[0]
    while tau + 1 <= max_lag and cmnd[tau + 1] < cmnd[tau]:
        tau += 1
    return samplerate / _refine_lag(cmnd, tau)


def pitch_frame_length(samplerate: int) -> int:
    """Samples per pitch frame: two periods of the lowest voice pitch, plus the lag search."""
    return 3 * int(samplerate / MIN_VOICE_HZ)


def _refine_lag(cmnd: np.ndarray, tau: int) -> float:
    """Parabolic interpolation around the minimum, for sub-sample period accuracy."""
    if tau <= 0 or tau >= len(cmnd) - 1:
        return float(tau)
    left, mid, right = cmnd[tau - 1], cmnd[tau], cmnd[tau + 1]
    curvature = left - 2 * mid + right
    return tau + (0.5 * (left - right) / curvature if curvature > 0 else 0.0)


def midi_note(frequency: float) -> float:
    """Fractional MIDI note number (A4 = 69) of a frequency."""
    return 69 + 12 * np.log2(frequency / 440)


def harmony_fit(note: float, profile: np.ndarray) -> float:
    """How well a sung note sits among the music's notes, from 0 to 1.

    Octave-blind: a note fits any pitch class it falls near, by how strongly
    the music plays that class, tapering off as the note strays from it.
    """
    offset = (np.arange(12) - note) % 12
    distance = np.minimum(offset, 12 - offset)
    return float(np.max(profile * np.exp(-0.5 * (distance / _PITCH_TOLERANCE) ** 2)))

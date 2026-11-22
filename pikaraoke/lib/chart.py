"""Note charts: the melody of a song as notes to sing, and how well a singer hits them.

A chart is what karaoke machines draw as bars to sing along to. Karaoke
videos carry none, so one is made from the original recording (see
chart_maker.py): its vocals are separated out and pitch-tracked, and the
pitch curve is cut into notes here. Charts are kept in the original
recording's time and shifted onto the video by the same offset as the lyrics.

Everything here is plain numpy, so it runs without the chart add-on.
"""

import json
import warnings
from dataclasses import dataclass

import numpy as np

CHART_VERSION = 1

# A pitch curve is cut into notes where the voice stops or moves to a new note.
# Each note is the median pitch it held, rounded to a semitone.
#
# Frames this quiet or unpitched are not singing.
_MIN_CONFIDENCE = 0.5
# A gap shorter than this inside a phrase is a consonant, not a new note.
_MAX_GAP_S = 0.06
# Moving this far from the note held, for this long, starts a new note.
# Vibrato swings about half a semitone either side, too quick to count.
_NOTE_CHANGE = 0.7
_NOTE_CHANGE_HOLD_S = 0.08
# Shorter notes are glides between notes or tracking errors.
_MIN_NOTE_S = 0.1
# The curve is smoothed over this window first, ironing out vibrato.
_SMOOTH_S = 0.07
# Adjacent notes at the same pitch this close together are one held note.
_MERGE_GAP_S = 0.08

# A sung note within this of the chart's, in semitones, hits it.
HIT_TOLERANCE = 1.0


@dataclass
class Note:
    start: float
    length: float
    pitch: int

    @property
    def end(self) -> float:
        return self.start + self.length


def notes_from_pitch(times: np.ndarray, midi: np.ndarray, confidence: np.ndarray) -> list[Note]:
    """Cut a pitch curve into notes.

    times are frame times in seconds at an even hop, midi each frame's pitch as
    a fractional MIDI note (NaN where there is none), confidence 0-1 how sure
    the tracker is that the frame is sung.
    """
    if len(times) < 2:
        return []
    hop = float(times[1] - times[0])
    voiced = (confidence >= _MIN_CONFIDENCE) & np.isfinite(midi)
    smooth = _smooth(midi, voiced, max(1, int(round(_SMOOTH_S / hop))))
    max_gap = int(round(_MAX_GAP_S / hop))
    hold = max(1, int(round(_NOTE_CHANGE_HOLD_S / hop)))

    notes: list[Note] = []
    current: list[int] = []  # frame indices of the note being sung
    drift = 0  # frames in a row away from the current note
    gap = 0

    def close() -> None:
        if current:
            pitch = float(np.median(smooth[current]))
            start, end = times[current[0]], times[current[-1]] + hop
            notes.append(Note(float(start), float(end - start), int(round(pitch))))

    for i in range(len(times)):
        if not voiced[i]:
            gap += 1
            if gap > max_gap and current:
                close()
                current, drift = [], 0
            continue
        gap = 0
        if not current:
            current = [i]
            continue
        held = np.median(smooth[current[-int(0.3 / hop) :]])
        if abs(smooth[i] - held) > _NOTE_CHANGE:
            drift += 1
            if drift >= hold:
                # The new note began where the drift did.
                moved = current[-(drift - 1) :] if drift > 1 else []
                del current[len(current) - len(moved) :]
                close()
                current, drift = moved + [i], 0
                continue
        else:
            drift = 0
        current.append(i)
    close()
    return _tidy(notes)


def _smooth(midi: np.ndarray, voiced: np.ndarray, width: int) -> np.ndarray:
    """A running median over voiced frames only, so silence does not drag notes down."""
    out = np.where(voiced, midi, np.nan)
    if width <= 1:
        return out
    half = width // 2
    padded = np.pad(out, half, constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1)
    with warnings.catch_warnings():
        # Every window centred on a voiced frame holds a value; the rest are discarded.
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(windows, axis=1)
    return np.where(voiced, med, np.nan)


def _tidy(notes: list[Note]) -> list[Note]:
    """Drop blips, then join repeats of a note with only a breath between."""
    out: list[Note] = []
    for note in notes:
        if note.length < _MIN_NOTE_S:
            continue
        if out and out[-1].pitch == note.pitch and note.start - out[-1].end <= _MERGE_GAP_S:
            out[-1].length = note.end - out[-1].start
            continue
        out.append(note)
    return out


def to_json(notes: list[Note], source: str, **extra) -> str:
    """The stored form: notes as [start, length, pitch] rows, in recording time."""
    return json.dumps(
        {
            "version": CHART_VERSION,
            "source": source,
            "notes": [[round(n.start, 3), round(n.length, 3), n.pitch] for n in notes],
            **extra,
        }
    )


def from_json(text: str | None) -> list[Note]:
    if not text:
        return []
    try:
        rows = json.loads(text).get("notes", [])
    except (ValueError, AttributeError):
        return []
    return [Note(float(t), float(d), int(p)) for t, d, p in rows]


def fold_octave(sung: float, target: float) -> float:
    """The sung note moved by octaves to sit nearest the target.

    A man singing a woman's melody an octave down is singing it right.
    """
    return sung + 12 * round((target - sung) / 12)


class MelodyIndex:
    """Looks up which chart note, if any, is sounding at a moment."""

    def __init__(self, notes: list[Note], offset: float = 0.0, semitones: int = 0) -> None:
        self.notes = notes
        self._starts = np.array([n.start + offset for n in notes])
        self._ends = np.array([n.end + offset for n in notes])
        self._pitches = np.array([n.pitch + semitones for n in notes], dtype=float)

    def __len__(self) -> int:
        return len(self.notes)

    def at(self, position: float) -> int | None:
        """Index of the note sounding at position, in video time, or None."""
        i = int(np.searchsorted(self._starts, position, side="right")) - 1
        if i >= 0 and position < self._ends[i]:
            return i
        return None

    def start(self, i: int) -> float:
        return float(self._starts[i])

    def pitch(self, i: int) -> float:
        return float(self._pitches[i])

    def length(self, i: int) -> float:
        return float(self._ends[i] - self._starts[i])

    def total_length(self) -> float:
        return float(np.sum(self._ends - self._starts))

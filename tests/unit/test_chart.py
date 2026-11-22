"""Tests for note charts: cutting a sung pitch curve into notes, and finding notes by time."""

import numpy as np
import pytest

from pikaraoke.lib import chart
from pikaraoke.lib.chart import MelodyIndex, Note

HOP = 0.01


def _curve(*segments, vibrato=0.0, seed=0):
    """A pitch curve from (seconds, MIDI note or None) segments; None is silence."""
    rng = np.random.default_rng(seed)
    midi, confidence = [], []
    for seconds, note in segments:
        n = int(round(seconds / HOP))
        if note is None:
            midi += [np.nan] * n
            confidence += [0.0] * n
        else:
            t = np.arange(n) * HOP
            midi += list(note + vibrato * np.sin(2 * np.pi * 5.5 * t) + rng.normal(0, 0.05, n))
            confidence += [0.9] * n
    times = np.arange(len(midi)) * HOP
    return times, np.array(midi), np.array(confidence)


def _summary(notes):
    return [(round(n.start, 2), round(n.length, 1), n.pitch) for n in notes]


def test_held_notes_become_notes():
    notes = chart.notes_from_pitch(*_curve((0.5, None), (0.5, 60), (0.5, 64), (0.3, None)))

    assert _summary(notes) == [(0.5, 0.5, 60), (1.0, 0.5, 64)]


def test_vibrato_stays_one_note():
    notes = chart.notes_from_pitch(*_curve((1.5, 67), vibrato=0.5))

    assert [n.pitch for n in notes] == [67]


def test_a_consonant_gap_does_not_split_a_note_but_a_breath_does():
    consonant = chart.notes_from_pitch(*_curve((0.4, 62), (0.04, None), (0.4, 62)))
    breath = chart.notes_from_pitch(*_curve((0.4, 62), (0.3, None), (0.4, 64)))

    assert [n.pitch for n in consonant] == [62]
    assert [n.pitch for n in breath] == [62, 64]


def test_blips_are_dropped():
    notes = chart.notes_from_pitch(*_curve((0.5, 60), (0.05, 72), (0.5, 60)))

    assert [n.pitch for n in notes] == [60]


def test_unsure_frames_are_not_singing():
    times, midi, confidence = _curve((1.0, 60))
    confidence[:] = 0.2

    assert chart.notes_from_pitch(times, midi, confidence) == []


def test_a_chart_survives_storage():
    notes = [Note(1.0, 0.5, 60), Note(1.6, 0.25, 62)]

    stored = chart.to_json(notes, "generated", original="https://y/original")

    assert chart.from_json(stored) == notes
    assert chart.from_json(None) == [] and chart.from_json("not json") == []


@pytest.mark.parametrize(
    "sung, target, folded", [(48.2, 60, 60.2), (71.0, 60, 59.0), (60.0, 60, 60.0), (54.0, 60, 54.0)]
)
def test_octaves_are_folded_onto_the_target(sung, target, folded):
    assert chart.fold_octave(sung, target) == pytest.approx(folded)


def test_melody_index_finds_the_note_sounding_in_video_time():
    melody = MelodyIndex([Note(1.0, 0.5, 60), Note(2.0, 1.0, 64)], offset=7.0, semitones=2)

    assert melody.at(8.2) == 0 and melody.pitch(0) == 62
    assert melody.at(8.6) is None
    assert melody.at(9.9) == 1 and melody.start(1) == 9.0
    assert melody.at(5.0) is None

"""Tests for the pitch and harmony analysis behind singing scores."""

import numpy as np
import pytest

from pikaraoke.lib import pitch


def _tone(midi: float, rate: int, seconds: float = 0.2) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    phase = 2 * np.pi * 440 * 2 ** ((midi - 69) / 12) * t
    return sum(np.sin(k * phase) / k for k in range(1, 9))


@pytest.mark.parametrize("rate", [16000, 48000])
@pytest.mark.parametrize("midi", [43, 57, 64, 76])
def test_detects_sung_pitch_within_a_few_cents(rate, midi):
    """Bass to soprano, at the scoring capture rate and the Windows stream rate."""
    frame = _tone(midi, rate)[: pitch.pitch_frame_length(rate)]

    assert pitch.midi_note(pitch.detect_pitch(frame, rate)) == pytest.approx(midi, abs=0.05)


def test_noise_has_no_pitch():
    rate = 16000
    noise = np.random.default_rng(0).normal(0, 1, pitch.pitch_frame_length(rate))

    assert pitch.detect_pitch(noise, rate) is None


def test_chroma_finds_the_notes_of_a_chord():
    rate = pitch.BACKING_SAMPLERATE
    c_major = sum(_tone(m, rate, seconds=1.0) for m in (48, 60, 64, 67))

    chroma = pitch.backing_chroma(c_major, rate)
    strongest = set(np.argsort(chroma.mean(axis=0))[-3:])

    assert strongest == {0, 4, 7}  # C, E, G


def test_a_note_in_the_chord_fits_and_a_clash_does_not():
    profile = np.zeros(12)
    profile[[0, 4, 7]] = 1.0  # C major

    assert pitch.harmony_fit(64.0, profile) == pytest.approx(1.0)  # E
    assert pitch.harmony_fit(76.1, profile) > 0.9  # E an octave up, slightly sharp
    assert pitch.harmony_fit(63.0, profile) < 0.05  # E flat clashes


def test_harmony_ignores_noise_that_lights_every_note():
    """Drums put energy in every pitch class; only notes standing out count."""
    drums_and_chord = np.full(12, 5.0)
    drums_and_chord[[0, 4, 7]] += 3.0

    profile = pitch.harmony_profile(drums_and_chord, drums_and_chord)

    assert profile[1] == 0 and profile[0] == pytest.approx(1.0)

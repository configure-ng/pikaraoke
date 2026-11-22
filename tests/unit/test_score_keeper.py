"""Tests for ScoreKeeper: real singing scores from mic audio against the song."""

import zipfile

import numpy as np
import pytest

from pikaraoke.lib import pitch
from pikaraoke.lib import score_keeper as sk
from pikaraoke.lib.events import EventSystem
from pikaraoke.lib.score_keeper import PlaybackState, ScoreKeeper

MIC_RATE = 16000
CHUNK = 800  # 50 ms, as the sound manager delivers it
# C, G, Am, F: two seconds each.
CHORDS = [[48, 60, 64, 67], [43, 59, 62, 67], [45, 60, 64, 69], [41, 60, 65, 69]]
SONG_S = 8.0
RNG = np.random.default_rng(3)


def _tone(midi: float, seconds: float, rate: int, harmonics: int = 6) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    phase = 2 * np.pi * 440 * 2 ** ((midi - 69) / 12) * t
    return sum(np.sin(k * phase) / k for k in range(1, harmonics + 1))


def _backing(semitones: int = 0) -> np.ndarray:
    rate = pitch.BACKING_SAMPLERATE
    x = np.concatenate([sum(_tone(m + semitones, 2.0, rate) for m in c) for c in CHORDS])
    return (x / np.abs(x).max()).astype(np.float32)


def _sing(notes_per_chord, semitones: int = 0, level: float = 0.3) -> np.ndarray:
    """Half-second notes with short breaths; notes_per_chord(chord) picks each note."""
    out = []
    for chord in CHORDS:
        for _ in range(4):
            note = _tone(notes_per_chord(chord) + semitones, 0.4, MIC_RATE, harmonics=8)
            out += [note / np.abs(note).max() * level, np.zeros(int(0.1 * MIC_RATE))]
    return np.concatenate(out) + RNG.normal(0, 0.002, int(SONG_S * MIC_RATE))


def in_tune(chord):
    return RNG.choice(chord[1:]) % 12 + 60


def off_key(chord):
    return RNG.uniform(55, 72)


def silence():
    return RNG.normal(0, 0.002, int(SONG_S * MIC_RATE))


class _Song:
    """A ScoreKeeper over a fake playback whose position the test drives."""

    def __init__(self, monkeypatch, semitones=0, enabled=True, backing=None):
        monkeypatch.setattr(
            sk, "decode_audio", lambda path: _backing() if backing is None else backing
        )
        self.events = EventSystem()
        self.state = PlaybackState("PB1", "/songs/song.mp4", semitones, None, False)
        self.captures = []
        self.keeper = ScoreKeeper(
            events=self.events,
            set_capture=self.captures.append,
            get_playback=lambda: self.state,
            is_enabled=lambda: enabled,
            spawn=lambda f, *args: f(*args),
        )

    def play(self, **mics: np.ndarray) -> dict | None:
        self.events.emit("playback_started")
        for start in range(0, int(SONG_S * MIC_RATE), CHUNK):
            self.state.position = (start + CHUNK) / MIC_RATE
            for mic, signal in mics.items():
                chunk = np.clip(signal[start : start + CHUNK], -1, 1)
                self.keeper.feed(mic, (chunk * 32767).astype(np.int16).tobytes(), MIC_RATE)
        return self.keeper.result("PB1")


def test_singing_in_tune_beats_singing_off_key(monkeypatch):
    good = _Song(monkeypatch).play(mic=_sing(in_tune))
    bad = _Song(monkeypatch).play(mic=_sing(off_key))

    assert good["mics"][0]["score"] >= 85
    assert bad["mics"][0]["score"] <= 50


def test_no_mic_means_no_score(monkeypatch):
    """Nothing to measure skips the score screen, rather than inventing a number."""
    assert _Song(monkeypatch).play() is None


def test_a_mic_on_but_unsung_scores_zero(monkeypatch):
    assert _Song(monkeypatch).play(mic=silence()) == {
        "mics": [{"number": 1, "score": 0}],
        "combined": None,
    }


def test_duet_scores_each_mic_and_their_average(monkeypatch):
    result = _Song(monkeypatch).play(a=_sing(in_tune), b=_sing(off_key))

    first, second = result["mics"]
    assert (first["number"], second["number"]) == (1, 2)
    assert first["score"] > second["score"]
    assert result["combined"] == round((first["score"] + second["score"]) / 2)


def test_an_unused_mic_leaves_the_duet_but_keeps_numbering(monkeypatch):
    """Mic 2 stays Mic 2 when only it was sung into."""
    result = _Song(monkeypatch).play(a=silence(), b=_sing(in_tune))

    assert [mic["number"] for mic in result["mics"]] == [2]
    assert result["combined"] is None


def test_singing_follows_a_key_change(monkeypatch):
    """A transposed song is heard shifted; following it is in tune, ignoring it is not."""
    followed = _Song(monkeypatch, semitones=2).play(mic=_sing(in_tune, semitones=2))
    ignored = _Song(monkeypatch, semitones=2).play(mic=_sing(in_tune))

    assert followed["mics"][0]["score"] >= 85
    assert ignored["mics"][0]["score"] < followed["mics"][0]["score"] - 30


def test_audio_while_paused_is_not_scored(monkeypatch):
    song = _Song(monkeypatch)
    song.events.emit("playback_started")
    song.state.position, song.state.is_paused = 1.0, True

    song.keeper.feed("mic", (_sing(in_tune)[:CHUNK] * 32767).astype(np.int16).tobytes(), MIC_RATE)

    assert song.keeper.result("PB1") == {"mics": [{"number": 1, "score": 0}], "combined": None}


def test_result_is_only_for_the_song_that_played(monkeypatch):
    song = _Song(monkeypatch)
    song.play(mic=_sing(in_tune))

    assert song.keeper.result("PB0") is None
    assert song.keeper.result(None) is None


def test_mics_are_captured_only_while_a_song_plays(monkeypatch):
    song = _Song(monkeypatch)
    song.events.emit("playback_started")
    song.events.emit("song_ended", "complete")

    assert song.captures == [song.keeper.feed, None]


def test_disabled_scoring_listens_to_nothing(monkeypatch):
    decoded = []
    song = _Song(monkeypatch, enabled=False)
    monkeypatch.setattr(sk, "decode_audio", lambda path: decoded.append(path))

    song.events.emit("playback_started")

    assert song.captures == []
    assert decoded == []


def test_an_undecodable_song_has_no_score(monkeypatch):
    monkeypatch.setattr(sk, "decode_audio", lambda path: None)
    song = _Song(monkeypatch)
    monkeypatch.setattr(sk, "decode_audio", lambda path: None)

    assert song.play(mic=_sing(in_tune)) is None


class TestDecodeAudio:
    @pytest.fixture
    def ffmpeg(self, monkeypatch):
        calls = []

        def fake_run(cmd, input=None, **kwargs):
            calls.append((cmd, input))
            return type("Done", (), {"returncode": 0, "stdout": b"\x00\x40" * 4, "stderr": b""})

        monkeypatch.setattr(sk.subprocess, "run", fake_run)
        return calls

    def test_decodes_a_song_file_to_mono_floats(self, ffmpeg):
        samples = sk.decode_audio("/songs/song.mp4")

        cmd, stdin = ffmpeg[0]
        assert cmd[:2] == ["ffmpeg", "-nostdin"] and "/songs/song.mp4" in cmd and stdin is None
        assert cmd[cmd.index("-ar") + 1] == str(pitch.BACKING_SAMPLERATE)
        assert np.allclose(samples, 0.5)

    def test_streams_the_mp3_out_of_a_zipped_cdg(self, ffmpeg, tmp_path):
        archive = tmp_path / "Song.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("Song.cdg", b"graphics")
            z.writestr("Song.MP3", b"mp3 bytes")

        sk.decode_audio(str(archive))

        cmd, stdin = ffmpeg[0]
        assert "-nostdin" not in cmd and cmd[cmd.index("-i") + 1] == "pipe:0"
        assert stdin == b"mp3 bytes"

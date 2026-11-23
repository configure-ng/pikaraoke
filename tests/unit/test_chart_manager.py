"""Tests for ChartManager: making each song's note chart once, in the background."""

import json

import pytest

from pikaraoke.lib import chart_manager as cm
from pikaraoke.lib.chart_manager import ChartManager
from pikaraoke.lib.events import EventSystem
from pikaraoke.lib.karaoke_database import KaraokeDatabase
from pikaraoke.lib.lyrics import Aligned

SONG = "/songs/夜に駆ける - YOASOBI---abc.mp4"


def _pitch_curve(path):
    """What chart_maker writes: a second at C4, a breath, half a second at E4."""
    midi = [60.0] * 100 + [None] * 30 + [64.0] * 50
    confidence = [0.9] * 100 + [0.0] * 30 + [0.9] * 50
    with open(path, "w") as f:
        json.dump({"hop": 0.01, "midi": midi, "confidence": confidence}, f)


class _Room:
    def __init__(self, tmp_path, original=None, located=None, makes=True, available=True):
        self.db = KaraokeDatabase(str(tmp_path / "pikaraoke.db"))
        self.db.insert_songs([{"file_path": SONG, "youtube_id": "abc", "format": "mp4"}])
        if original:
            self.db.set_original(SONG, original, 7.5)
        self.events = EventSystem()
        self.playing = None
        self.notified = 0
        self.located, self.downloaded, self.made = [], [], []
        self.enabled = True

        def locate(song, query, duration):
            self.located.append(query)
            return located

        def download(url, dest):
            self.downloaded.append(url)
            path = f"{dest}/original.webm"
            open(path, "w").close()
            return path

        def make_pitch(audio, out, model_dir):
            self.made.append(model_dir)
            if makes:
                _pitch_curve(out)
            return makes

        self.manager = ChartManager(
            db=self.db,
            events=self.events,
            model_dir="/models",
            get_now_playing=lambda: self.playing,
            get_query=lambda path, row: "夜に駆ける YOASOBI",
            get_duration=lambda path: 270,
            notify=self._notify,
            is_enabled=lambda: self.enabled,
            available=lambda: available,
            locate=locate,
            download=download,
            make_pitch=make_pitch,
            spawn=lambda f, *args: f(*args),
        )

    def _notify(self):
        self.notified += 1


@pytest.fixture
def room(tmp_path):
    return _Room(tmp_path, original="https://y/original")


def test_a_chart_is_made_from_the_original_the_lyrics_found(room):
    room.events.emit("lyrics_looked_up", SONG)

    assert room.downloaded == ["https://y/original"] and room.made == ["/models"]
    assert room.located == []
    assert room.manager.status(SONG) == "found"
    assert room.manager.notes(SONG) == [[0.0, 1.0, 60], [1.3, 0.5, 64]]


def test_the_chart_scores_in_video_time(room):
    room.manager.request(SONG)

    melody = room.manager.melody(SONG)
    assert melody.offset == 7.5 and [n.pitch for n in melody.notes] == [60, 64]


def test_without_an_original_one_is_searched_for(tmp_path):
    room = _Room(tmp_path, located=Aligned("https://y/found", 3.0))

    room.manager.request(SONG)

    assert room.located == ["夜に駆ける YOASOBI"] and room.downloaded == ["https://y/found"]
    assert room.db.get_chart(SONG)["original_url"] == "https://y/found"
    assert room.manager.melody(SONG).offset == 3.0


def test_no_original_anywhere_is_missing_and_can_be_retried(tmp_path):
    room = _Room(tmp_path)

    room.manager.request(SONG)
    assert room.manager.status(SONG) == "missing" and room.downloaded == []

    room.manager.request(SONG)
    assert len(room.located) == 1  # not tried again by itself

    room.manager.retry(SONG)
    assert len(room.located) == 2


def test_a_failed_make_is_recorded(tmp_path):
    room = _Room(tmp_path, original="https://y/original", makes=False)

    room.manager.request(SONG)

    assert room.manager.status(SONG) == "failed" and room.manager.notes(SONG) is None


def test_nothing_is_made_without_the_add_on_or_when_turned_off(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    without = _Room(tmp_path / "a", original="https://y/original", available=False)
    without.manager.request(SONG)
    assert without.made == [] and without.manager.status(SONG) is None

    off = _Room(tmp_path / "b", original="https://y/original")
    off.enabled = False
    off.manager.request(SONG)
    assert off.made == []


def test_the_playing_song_is_told_when_its_chart_changes(room):
    room.playing = SONG

    room.manager.request(SONG)

    assert room.notified == 2  # making, then made


def test_a_played_song_already_looked_up_gets_its_chart(room):
    room.db.set_lyrics(SONG, None)
    room.playing = SONG

    room.events.emit("playback_started")

    assert room.manager.status(SONG) == "found"


def test_a_played_song_not_yet_looked_up_waits_for_the_lookup(room):
    room.playing = SONG

    room.events.emit("playback_started")

    assert room.made == []


def test_the_whole_library_can_be_queued(room):
    assert room.manager.request_all() == 0  # made at once with an inline spawn
    assert room.manager.status(SONG) == "found"


def test_the_add_on_check_is_cheap_and_cached(tmp_path):
    calls = []
    room = _Room(tmp_path)
    room.manager._available = lambda: calls.append(1) or True

    assert room.manager.available and room.manager.available
    assert calls == [1]


def test_reading_a_pitch_curve(tmp_path):
    path = tmp_path / "pitch.json"
    _pitch_curve(path)

    notes = cm.notes_from_file(str(path))

    assert [(n.pitch, round(n.length, 1)) for n in notes] == [(60, 1.0), (64, 0.5)]


def test_a_refused_download_is_tried_again(monkeypatch):
    answers = iter([None, None, "/tmp/original.webm"])
    monkeypatch.setattr(cm, "download_audio", lambda *a, **k: next(answers))
    waits = []

    assert (
        cm._download_original("https://y/original", "/tmp", wait=waits.append)
        == "/tmp/original.webm"
    )
    assert len(waits) == 2


def test_a_download_refused_every_time_gives_up(monkeypatch):
    calls = []
    monkeypatch.setattr(cm, "download_audio", lambda *a, **k: calls.append(1))

    assert cm._download_original("https://y/original", "/tmp", wait=lambda s: None) is None
    assert len(calls) == cm._DOWNLOAD_ATTEMPTS

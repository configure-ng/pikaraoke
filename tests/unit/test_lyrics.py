"""Tests for finding, aligning and serving generated lyrics."""

import json

import numpy as np
import pytest

from pikaraoke.lib import lyrics, pitch
from pikaraoke.lib.events import EventSystem
from pikaraoke.lib.karaoke_database import KaraokeDatabase
from pikaraoke.lib.lyrics import Found, LyricsManager
from pikaraoke.lib.youtube_dl import SearchResult

SONG = "/songs/夜に駆ける - YOASOBI---1ZWHwwtxB3s.mp4"
OTHER = "/songs/Lemon - 米津玄師---aaaaaaaaaaa.mp4"


def _entry(track, artist="YOASOBI", duration=262.0, lrc="[00:10.00]沈むように", **extra):
    return {
        "id": 1,
        "trackName": track,
        "artistName": artist,
        "duration": duration,
        "syncedLyrics": lrc,
        **extra,
    }


class TestParseLrc:
    def test_reads_timed_lines_in_order_and_skips_tags(self):
        lrc = "[ar:YOASOBI]\n[00:15.50]二人だけの\n[00:09.01] 沈むように \n[01:02.5]"

        assert lyrics.parse_lrc(lrc) == [(9.01, "沈むように"), (15.5, "二人だけの"), (62.5, "")]

    def test_a_repeated_line_shows_at_each_of_its_times(self):
        assert lyrics.parse_lrc("[00:01.00][00:30.00]chorus") == [(1.0, "chorus"), (30.0, "chorus")]

    def test_a_translation_under_a_line_is_dropped(self):
        lrc = "[00:05.00]夜に駆ける\n[00:05.00]Racing into the night"

        assert lyrics.parse_lrc(lrc) == [(5.0, "夜に駆ける")]


def test_japanese_lyrics_carry_romaji_and_kana():
    built = lyrics.build_lyrics([(1.0, "君は僕に似ている"), (4.0, "")])

    assert built["japanese"] is True
    assert built["lines"][0] == {
        "t": 1.0,
        "text": "君は僕に似ている",
        "romaji": "Kimi wa boku ni nite iru",
        "kana": "きみはぼくににている",
    }
    assert built["lines"][1] == {"t": 4.0, "text": "", "romaji": "", "kana": ""}


def test_other_lyrics_are_kept_as_written():
    assert lyrics.build_lyrics([(1.0, "Hello")]) == {
        "japanese": False,
        "lines": [{"t": 1.0, "text": "Hello"}],
    }


class TestPickEntry:
    def test_prefers_the_entry_naming_the_artist_too(self):
        results = [_entry("夜に駆ける", artist="Cover Singer"), _entry("夜に駆ける")]

        assert lyrics.pick_entry(results, "夜に駆ける - YOASOBI", None)["artistName"] == "YOASOBI"

    def test_skips_karaoke_and_instrumental_cuts(self):
        results = [_entry("YOASOBI 夜に駆ける(inst)"), _entry("夜に駆ける", duration=275.0)]

        assert lyrics.pick_entry(results, "夜に駆ける - YOASOBI", None)["duration"] == 275.0

    def test_among_equals_takes_the_nearest_length(self):
        results = [_entry("夜に駆ける", duration=200.0), _entry("夜に駆ける", duration=268.0)]

        assert lyrics.pick_entry(results, "夜に駆ける - YOASOBI", 270)["duration"] == 268.0

    def test_a_different_song_is_no_match(self):
        assert lyrics.pick_entry([_entry("アイドル")], "夜に駆ける - YOASOBI", None) is None


def _song_chroma(seconds: float, seed: int) -> np.ndarray:
    """Chroma of a song: a new chord every half second."""
    rng = np.random.default_rng(seed)
    rows = int(seconds / pitch.CHROMA_HOP_S)
    chords = rng.integers(0, 12, size=(rows // 5 + 1, 3))
    chroma = rng.uniform(0, 0.2, size=(rows, 12))
    for row in range(rows):
        chroma[row, chords[row // 5]] += 1.0
    return chroma


class TestAlignmentOffset:
    def test_finds_how_far_the_karaoke_video_starts_behind(self):
        original = _song_chroma(120, seed=1)
        intro = np.random.default_rng(2).uniform(0, 0.2, size=(72, 12))  # 7.2 s title card
        karaoke = np.vstack([intro, original])[:1100]

        assert lyrics.alignment_offset(karaoke, original) == pytest.approx(7.2)

    def test_finds_it_in_a_different_key(self):
        original = _song_chroma(120, seed=1)
        karaoke = np.roll(original[30:], 2, axis=1)  # starts 3 s in, two semitones up

        assert lyrics.alignment_offset(karaoke, original) == pytest.approx(-3.0)

    def test_a_different_song_does_not_align(self):
        assert lyrics.alignment_offset(_song_chroma(120, 1), _song_chroma(120, 5)) is None


def test_the_original_is_the_result_matching_the_lyrics_length(monkeypatch):
    monkeypatch.setattr(
        lyrics,
        "get_search_results",
        lambda query: [
            SearchResult("夜に駆ける カラオケ", "https://y/karaoke", "k", "", "4:22"),
            SearchResult("夜に駆ける (live)", "https://y/live", "l", "", "5:10"),
            SearchResult("夜に駆ける", "https://y/original", "o", "", "4:23"),
        ],
    )

    assert lyrics.find_original(_entry("夜に駆ける", duration=262.0)) == ["https://y/original"]


def test_an_original_that_will_not_download_gives_way_to_the_next(monkeypatch):
    monkeypatch.setattr(
        lyrics, "find_original", lambda entry: ["https://y/age-gated", "https://y/topic"]
    )
    monkeypatch.setattr(
        lyrics, "align_to", lambda song, url, run: None if "age-gated" in url else 7.5
    )

    aligned = lyrics.measure_offset("/songs/a.mp4", _entry("夜に駆ける"))

    assert aligned == lyrics.Aligned("https://y/topic", 7.5)


class TestLocateOriginal:
    """For songs with no lyrics entry: search by name, keep what lines up."""

    @pytest.fixture
    def results(self, monkeypatch):
        found = [
            SearchResult("夜に駆ける カラオケ", "https://y/karaoke", "k", "", "4:22"),
            SearchResult("夜に駆ける cover", "https://y/cover", "c", "", "4:25"),
            SearchResult("夜に駆ける", "https://y/original", "o", "", "4:23"),
            SearchResult("夜に駆ける 1 hour loop", "https://y/loop", "h", "", "60:00"),
        ]
        monkeypatch.setattr(lyrics, "get_search_results", lambda query: found)
        tried = []

        def align(song, url, run):
            tried.append(url)
            return 7.0 if url == "https://y/original" else None

        monkeypatch.setattr(lyrics, "align_to", align)
        return tried

    def test_the_first_result_that_lines_up_is_taken(self, results):
        aligned = lyrics.locate_original("/songs/a.mp4", "夜に駆ける YOASOBI", 270)

        assert aligned == lyrics.Aligned("https://y/original", 7.0)
        # Karaoke versions are never tried; covers are, and do not line up.
        assert results == ["https://y/cover", "https://y/original"]

    def test_results_far_from_the_video_length_are_skipped(self, results):
        lyrics.locate_original("/songs/a.mp4", "夜に駆ける YOASOBI", 270)

        assert "https://y/loop" not in results

    def test_nothing_lining_up_is_nothing_found(self, monkeypatch, results):
        monkeypatch.setattr(lyrics, "align_to", lambda song, url, run: None)

        assert lyrics.locate_original("/songs/a.mp4", "夜に駆ける YOASOBI", 270) is None


class TestFindLyrics:
    @pytest.fixture
    def online(self, monkeypatch):
        lrc = "[00:10.00]二人だけの空\n[00:20.00]"
        monkeypatch.setattr(lyrics, "search_lrclib", lambda q: [_entry("夜に駆ける", lrc=lrc)])
        offsets = {"value": 7.5}
        monkeypatch.setattr(
            lyrics,
            "measure_offset",
            lambda path, entry, run: (
                None
                if offsets["value"] is None
                else lyrics.Aligned("https://y/original", offsets["value"])
            ),
        )
        return offsets

    def test_found_lyrics_are_stored_with_their_offset(self, online):
        found = lyrics.find_lyrics(SONG, "夜に駆ける - YOASOBI", 270)

        stored = json.loads(found.lyrics)
        assert found.offset == 7.5 and stored["aligned"] is True
        assert found.original == "https://y/original"
        assert stored["lines"][0]["romaji"] == "Futari dake no sora"

    def test_unaligned_lyrics_keep_the_recording_timing(self, online):
        online["value"] = None

        found = lyrics.find_lyrics(SONG, "夜に駆ける - YOASOBI", 270)

        assert found.offset == 0.0 and json.loads(found.lyrics)["aligned"] is False

    def test_nothing_found_is_none(self, monkeypatch):
        monkeypatch.setattr(lyrics, "search_lrclib", lambda q: [])

        assert lyrics.find_lyrics(SONG, "unknown", None) is None


class _Room:
    """A LyricsManager over a real database, with the lookup faked."""

    def __init__(self, tmp_path, found=None, playing=SONG):
        self.db = KaraokeDatabase(str(tmp_path / "test.db"))
        self.db.insert_songs(
            [{"file_path": p, "youtube_id": None, "format": "mp4"} for p in (SONG, OTHER)]
        )
        self.events = EventSystem()
        self.playing = playing
        self.lookups = []
        self.notified = 0
        default = Found(
            lyrics=json.dumps({"japanese": True, "lines": [{"t": 1, "text": "夜"}]}), offset=7.5
        )
        self.found = default if found is None else found

        def find(path, query, duration, run):
            self.lookups.append((path, query))
            return self.found or None

        self.spawned = []
        self.manager = LyricsManager(
            db=self.db,
            events=self.events,
            get_now_playing=lambda: self.playing,
            get_query=lambda path, row: f"query for {path}",
            get_duration=lambda path: 270,
            notify=self._notify,
            find=find,
            spawn=lambda f, *args: self.spawned.append((f, args)),
            run=lambda f, *args: f(*args),
        )

    def _notify(self):
        self.notified += 1

    def run_lookups(self):
        while self.spawned:
            f, args = self.spawned.pop(0)
            f(*args)


def test_a_song_is_looked_up_when_it_starts(tmp_path):
    room = _Room(tmp_path)
    room.events.emit("playback_started")

    assert room.manager.state(SONG)["status"] == "searching"
    room.run_lookups()

    assert room.lookups == [(SONG, f"query for {SONG}")]
    assert room.manager.state(SONG) == {
        "status": "found",
        "modes": ["off", "romaji", "kana"],
        "mode": "off",
        "offset": 7.5,
    }
    assert room.notified == 2  # searching, then found


def test_a_song_is_looked_up_only_once(tmp_path):
    room = _Room(tmp_path)
    room.manager.request(SONG)
    room.manager.request(SONG)
    room.run_lookups()
    room.manager.request(SONG)
    room.run_lookups()

    assert len(room.lookups) == 1


def test_lookups_run_one_at_a_time_playing_song_first(tmp_path):
    room = _Room(tmp_path)
    room.manager.request(OTHER)
    room.manager.request(SONG)
    room.run_lookups()

    assert [path for path, _ in room.lookups] == [SONG, OTHER]


def test_nothing_found_is_remembered_and_can_be_retried(tmp_path):
    room = _Room(tmp_path, found=False)
    room.manager.request(SONG)
    room.run_lookups()
    assert room.manager.state(SONG)["status"] == "missing"

    room.found = Found(lyrics=json.dumps({"japanese": False, "lines": []}), offset=0)
    room.manager.retry(SONG)
    room.run_lookups()

    assert room.manager.state(SONG)["status"] == "found"
    assert room.manager.state(SONG)["modes"] == ["off", "on"]


def test_mode_and_timing_are_remembered_per_song(tmp_path):
    room = _Room(tmp_path)
    room.manager.request(SONG)
    room.run_lookups()

    assert room.manager.set_mode(SONG, "kana")
    assert room.manager.nudge(SONG, -0.5)
    assert not room.manager.set_mode(SONG, "on")  # not a mode a Japanese song has

    assert room.manager.state(SONG)["mode"] == "kana"
    assert room.manager.state(SONG)["offset"] == 7.0
    assert room.manager.lines(SONG) == {"japanese": True, "lines": [{"t": 1, "text": "夜"}]}


def test_a_song_without_lyrics_serves_none(tmp_path):
    room = _Room(tmp_path)

    assert room.manager.lines(SONG) is None
    assert room.manager.state("/not/in/library.mp4") is None


def test_one_failing_lookup_does_not_stop_the_rest(tmp_path):
    room = _Room(tmp_path)
    original = room.manager._find

    def flaky(path, *args):
        if path == SONG:
            raise RuntimeError("network down")
        return original(path, *args)

    room.manager._find = flaky
    room.manager.request(SONG)
    room.manager.request(OTHER)
    room.run_lookups()

    assert room.manager.state(OTHER)["status"] == "found"
    assert room.manager.state(SONG)["status"] is None  # still unlooked, retried next play

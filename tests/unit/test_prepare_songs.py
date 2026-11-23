"""Tests for getting songs ready to sing ahead of time: lyrics, then note chart."""

from unittest.mock import MagicMock

from pikaraoke.karaoke import Karaoke


def _karaoke(lyrics_status):
    k = MagicMock()
    k.db.get_lyrics.return_value = {"status": lyrics_status}
    return k


def test_a_song_never_looked_up_gets_its_lyrics_first():
    k = _karaoke(None)

    Karaoke.prepare_song(k, "/songs/a.mp4")

    k.lyrics_manager.request.assert_called_once_with("/songs/a.mp4")
    k.chart_manager.request.assert_not_called()


def test_a_looked_up_song_gets_its_chart():
    k = _karaoke("found")

    Karaoke.prepare_song(k, "/songs/a.mp4")

    k.chart_manager.request.assert_called_once_with("/songs/a.mp4")
    k.chart_manager.retry_failed.assert_called_once_with("/songs/a.mp4")
    k.lyrics_manager.request.assert_not_called()


def test_the_whole_library_is_prepared():
    k = MagicMock()
    k.db.get_all_song_paths.return_value = ["/songs/a.mp4", "/songs/b.mp4"]

    Karaoke.prepare_library(k)

    assert [c.args[0] for c in k.prepare_song.call_args_list] == ["/songs/a.mp4", "/songs/b.mp4"]


def test_queue_position():
    k = MagicMock()
    k.queue_manager.queue = [{"file": "/songs/a.mp4"}, {"file": "/songs/b.mp4"}]

    assert Karaoke._queue_position(k, "/songs/b.mp4") == 1
    assert Karaoke._queue_position(k, "/songs/c.mp4") is None

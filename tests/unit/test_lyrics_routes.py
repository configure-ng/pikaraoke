"""Tests for the lyrics routes: they act only on the song playing now."""

from unittest.mock import MagicMock, patch

import pytest

from pikaraoke.routes.lyrics import lyrics_bp
from tests.conftest import make_route_app

PLAYING = "/songs/夜に駆ける---1ZWHwwtxB3s.mp4"


@pytest.fixture
def k():
    k = MagicMock()
    k.playback_controller.now_playing_filename = PLAYING
    with patch("pikaraoke.routes.lyrics.get_karaoke_instance", return_value=k):
        yield k


@pytest.fixture
def client():
    # Not an admin: the room's guests choose how lyrics show.
    return make_route_app(lyrics_bp, [], admin=False).test_client()


def test_serves_the_playing_songs_lines(k, client):
    k.lyrics_manager.lines.return_value = {"japanese": True, "lines": []}

    assert client.get("/api/lyrics").get_json() == {"japanese": True, "lines": []}
    k.lyrics_manager.lines.assert_called_once_with(PLAYING)


def test_sets_the_mode_of_the_playing_song(k, client):
    k.lyrics_manager.set_mode.return_value = True

    assert client.post("/api/lyrics/mode/romaji").status_code == 200
    k.lyrics_manager.set_mode.assert_called_once_with(PLAYING, "romaji")


def test_refuses_a_mode_the_song_does_not_have(k, client):
    k.lyrics_manager.set_mode.return_value = False

    assert client.post("/api/lyrics/mode/klingon").status_code == 400


def test_nothing_playing_changes_nothing(k, client):
    k.playback_controller.now_playing_filename = None

    assert client.post("/api/lyrics/mode/romaji").status_code == 400
    assert client.post("/api/lyrics/nudge?seconds=0.5").status_code == 400
    k.lyrics_manager.set_mode.assert_not_called()
    k.lyrics_manager.nudge.assert_not_called()


@pytest.mark.parametrize("seconds", ["0.5", "-0.5"])
def test_nudges_the_timing(k, client, seconds):
    k.lyrics_manager.nudge.return_value = True

    assert client.post(f"/api/lyrics/nudge?seconds={seconds}").status_code == 200
    k.lyrics_manager.nudge.assert_called_once_with(PLAYING, float(seconds))


@pytest.mark.parametrize("seconds", ["", "soon", "600", "nan"])
def test_refuses_a_nudge_that_is_not_a_small_number(k, client, seconds):
    assert client.post(f"/api/lyrics/nudge?seconds={seconds}").status_code == 400
    k.lyrics_manager.nudge.assert_not_called()


def test_retry_looks_again(k, client):
    client.post("/api/lyrics/retry")

    k.lyrics_manager.retry.assert_called_once_with(PLAYING)

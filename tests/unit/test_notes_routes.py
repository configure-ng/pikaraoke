"""Tests for the note chart routes: the room toggles the guide, the host makes charts."""

from unittest.mock import MagicMock, patch

import pytest

from pikaraoke.routes.notes import notes_bp
from tests.conftest import make_route_app

PLAYING = "/songs/夜に駆ける---1ZWHwwtxB3s.mp4"


@pytest.fixture
def k():
    k = MagicMock()
    k.playback_controller.now_playing_filename = PLAYING
    with (
        patch("pikaraoke.routes.notes.get_karaoke_instance", return_value=k),
        patch("pikaraoke.routes.notes.broadcast_event") as broadcast,
    ):
        k.broadcast = broadcast
        yield k


@pytest.fixture
def guest():
    return make_route_app(notes_bp, [], admin=False).test_client()


@pytest.fixture
def host():
    return make_route_app(notes_bp, [], admin=True).test_client()


def test_serves_the_playing_songs_chart(k, guest):
    k.chart_manager.notes.return_value = [[1.0, 0.5, 60]]

    assert guest.get("/api/notes").get_json() == [[1.0, 0.5, 60]]
    k.chart_manager.notes.assert_called_once_with(PLAYING)


@pytest.mark.parametrize("state, value", [("on", True), ("off", False)])
def test_anyone_turns_the_guide_on_or_off(k, guest, state, value):
    assert guest.post(f"/api/notes/guide/{state}").status_code == 200

    k.preferences.set.assert_called_once_with("note_guide", value)
    k.broadcast.assert_called_once_with("preferences_update", {"key": "note_guide", "value": value})
    k.update_now_playing_socket.assert_called_once()


def test_refuses_a_guide_state_that_is_not_on_or_off(k, guest):
    assert guest.post("/api/notes/guide/maybe").status_code == 400
    k.preferences.set.assert_not_called()


def test_retries_the_playing_songs_chart(k, guest):
    assert guest.post("/api/notes/retry").get_json() == {"success": True}
    k.chart_manager.retry.assert_called_once_with(PLAYING)


def test_making_the_whole_library_is_for_the_host(k, host):
    k.chart_manager.available = True
    k.make_all_charts.return_value = 12

    assert host.post("/api/notes/make_all").get_json() == {"success": True, "queued": 12}


def test_making_charts_needs_the_add_on(k, host):
    k.chart_manager.available = False

    assert host.post("/api/notes/make_all").status_code == 400
    k.make_all_charts.assert_not_called()


def test_status_reports_the_add_on_and_the_work(k, host):
    k.chart_manager.available = True
    k.chart_manager.progress.return_value = {"making": PLAYING, "queued": 3}
    k.preferences.get_or_default.return_value = True

    assert host.get("/api/notes/status").get_json() == {
        "available": True,
        "enabled": True,
        "making": PLAYING,
        "queued": 3,
    }

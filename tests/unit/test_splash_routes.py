"""Tests for splash routes — score phrase helpers and endpoint."""

import re
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask
from flask_babel import Babel

from pikaraoke.routes.splash import (
    _default_score_phrases,
    _get_active_score_phrases,
    splash_bp,
)
from tests.conftest import make_route_app


@pytest.fixture
def app():
    test_app = Flask(__name__)
    Babel(test_app)
    test_app.register_blueprint(splash_bp)
    return test_app


@pytest.fixture
def app_ctx(app):
    """Provide a Flask app context for tests that call helpers directly."""
    with app.app_context():
        yield


@pytest.fixture
def client(app):
    return app.test_client()


def _make_karaoke(low="", mid="", high=""):
    k = MagicMock()
    k.low_score_phrases = low
    k.mid_score_phrases = mid
    k.high_score_phrases = high
    return k


class TestDefaultScorePhrases:
    """Tests for _default_score_phrases()."""

    def test_returns_all_tiers(self, app_ctx):
        phrases = _default_score_phrases()
        assert set(phrases.keys()) == {"low", "mid", "high"}

    def test_each_tier_has_phrases(self, app_ctx):
        phrases = _default_score_phrases()
        for tier in ("low", "mid", "high"):
            assert len(phrases[tier]) > 0
            assert all(isinstance(p, str) for p in phrases[tier])


class TestGetActiveScorePhrases:
    """Tests for _get_active_score_phrases()."""

    def test_returns_defaults_when_no_custom_phrases(self, app_ctx):
        result = _get_active_score_phrases(_make_karaoke())
        assert result == _default_score_phrases()

    def test_returns_custom_phrases_with_pipe_separator(self, app_ctx):
        k = _make_karaoke(low="Bad|Terrible", mid="OK|Alright", high="Great|Amazing")
        result = _get_active_score_phrases(k)
        assert result["low"] == ["Bad", "Terrible"]
        assert result["mid"] == ["OK", "Alright"]
        assert result["high"] == ["Great", "Amazing"]

    def test_handles_legacy_newline_separator(self, app_ctx):
        result = _get_active_score_phrases(_make_karaoke(low="Bad\nTerrible"))
        assert result["low"] == ["Bad", "Terrible"]

    def test_falls_back_to_defaults_when_all_whitespace(self, app_ctx):
        result = _get_active_score_phrases(_make_karaoke(low="   |  |  "))
        assert result["low"] == _default_score_phrases()["low"]

    def test_mixed_custom_and_default(self, app_ctx):
        result = _get_active_score_phrases(_make_karaoke(low="Custom low", high="Custom high"))
        defaults = _default_score_phrases()
        assert result["low"] == ["Custom low"]
        assert result["mid"] == defaults["mid"]
        assert result["high"] == ["Custom high"]


class TestScorePhrasesEndpoint:
    """Tests for GET /splash/score_phrases."""

    @patch("pikaraoke.routes.splash.get_karaoke_instance")
    def test_returns_json_with_all_tiers(self, mock_get_instance, client):
        mock_get_instance.return_value = _make_karaoke(low="Bad|Terrible")

        response = client.get("/api/splash/score_phrases")

        assert response.status_code == 200
        data = response.get_json()
        assert set(data.keys()) == {"low", "mid", "high"}
        assert data["low"] == ["Bad", "Terrible"]


class TestHidingTheUrlAndQrCode:
    """The URL and the QR code hide separately, in the corner and the screensaver."""

    @pytest.fixture
    def app(self):
        return make_route_app(
            splash_bp,
            [
                ("/logo", "images.logo"),
                ("/qrcode", "images.qrcode"),
                ("/api/lyrics", "lyrics.get_lyrics"),
            ],
        )

    def _hidden(self, client, css_class, **hidden):
        """Whether each element carrying `css_class` is rendered hidden."""
        k = MagicMock(is_raspberry_pi=False, bg_video_path=None, hide_url=False, hide_qr_code=False)
        k.configure_mock(**hidden)
        with (
            patch("pikaraoke.routes.splash.get_karaoke_instance", return_value=k),
            patch("pikaraoke.routes.splash.get_site_name", return_value="PiKaraoke"),
        ):
            page = client.get("/splash").get_data(as_text=True)
        tags = re.findall(rf'<[^<]*class="{css_class}[^<]*>', page)
        return ["display: none" in tag for tag in tags]

    def test_hiding_the_url_leaves_the_qr_code(self, client):
        assert self._hidden(client, "splash-url", hide_url=True) == [True, True]
        assert self._hidden(client, "splash-qr-image", hide_url=True) == [False, False]

    def test_hiding_the_qr_code_leaves_the_url(self, client):
        assert self._hidden(client, "splash-qr-image", hide_qr_code=True) == [True, True]
        assert self._hidden(client, "splash-url", hide_qr_code=True) == [False, False]


class TestScoreEndpoint:
    """GET /api/splash/score: the real scores the score screen shows."""

    @patch("pikaraoke.routes.splash.get_karaoke_instance")
    def test_names_each_mic_by_its_number(self, mock_get_instance, client):
        k = MagicMock()
        k.score_keeper.result.return_value = {
            "mics": [{"number": 1, "score": 82}, {"number": 2, "score": 64}],
            "combined": 73,
        }
        mock_get_instance.return_value = k

        response = client.get("/api/splash/score?playback_id=PB1")

        k.score_keeper.result.assert_called_once_with("PB1")
        assert response.get_json() == {
            "mics": [{"name": "Mic 1", "score": 82}, {"name": "Mic 2", "score": 64}],
            "combined": 73,
        }

    @patch("pikaraoke.routes.splash.get_karaoke_instance")
    def test_null_when_there_is_nothing_to_score(self, mock_get_instance, client):
        mock_get_instance.return_value.score_keeper.result.return_value = None

        response = client.get("/api/splash/score?playback_id=PB1")

        assert response.status_code == 200 and response.get_json() is None

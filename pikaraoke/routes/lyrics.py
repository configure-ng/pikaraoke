"""Lyrics for the playing song: the TV fetches them, any phone picks how they show.

Open to everyone in the room, like the song list: turning lyrics on or nudging
their timing changes what the TV shows, not what plays. Every route acts on the
song playing now, so no client names a file.
"""

from flask import jsonify, request
from flask_smorest import Blueprint

from pikaraoke.lib.auth import public
from pikaraoke.lib.current_app import get_karaoke_instance

lyrics_bp = Blueprint("lyrics", __name__)

# One nudge at most, so a stray request cannot throw the timing far off.
_MAX_NUDGE_S = 5.0


def _playing() -> str | None:
    return get_karaoke_instance().playback_controller.now_playing_filename


@lyrics_bp.route("/api/lyrics")
@public
def get_lyrics():
    """The playing song's timed lines, with romaji and kana for Japanese; null if none."""
    k = get_karaoke_instance()
    return jsonify(k.lyrics_manager.lines(_playing()))


@lyrics_bp.route("/api/lyrics/mode/<mode>", methods=["POST"])
@public
def set_mode(mode: str):
    """Show the playing song's lyrics as one of the modes its state lists."""
    song = _playing()
    ok = bool(song) and get_karaoke_instance().lyrics_manager.set_mode(song, mode)
    return jsonify({"success": ok}), 200 if ok else 400


@lyrics_bp.route("/api/lyrics/nudge", methods=["POST"])
@public
def nudge():
    """Move the playing song's lyrics later (positive seconds) or earlier."""
    try:
        seconds = float(request.args.get("seconds", ""))
    except ValueError:
        return jsonify({"success": False}), 400
    if not -_MAX_NUDGE_S <= seconds <= _MAX_NUDGE_S:
        return jsonify({"success": False}), 400
    song = _playing()
    ok = bool(song) and get_karaoke_instance().lyrics_manager.nudge(song, seconds)
    return jsonify({"success": ok}), 200 if ok else 400


@lyrics_bp.route("/api/lyrics/retry", methods=["POST"])
@public
def retry():
    """Look again for the playing song's lyrics, after a lookup found none."""
    song = _playing()
    if song:
        get_karaoke_instance().lyrics_manager.retry(song)
    return jsonify({"success": bool(song)})

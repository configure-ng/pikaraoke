"""Note charts: the TV fetches the playing song's, anyone turns the note guide on or off.

The guide is open to the whole room, like lyrics. Making charts for the whole
library, and what the add-on is doing, are for the host.
"""

from flask import jsonify
from flask_smorest import Blueprint

from pikaraoke.lib.auth import public
from pikaraoke.lib.current_app import broadcast_event, get_karaoke_instance

notes_bp = Blueprint("notes", __name__)


def _playing() -> str | None:
    return get_karaoke_instance().playback_controller.now_playing_filename


@notes_bp.route("/api/notes")
@public
def get_notes():
    """The playing song's chart as [start, length, MIDI note] rows, in the time
    of the recording it was made from (add the lyrics offset); null if none."""
    return jsonify(get_karaoke_instance().chart_manager.notes(_playing()))


@notes_bp.route("/api/notes/guide/<state>", methods=["POST"])
@public
def set_guide(state: str):
    """Show (on) or hide (off) the note guide on the TV, for every song."""
    if state not in ("on", "off"):
        return jsonify({"success": False}), 400
    k = get_karaoke_instance()
    value = state == "on"
    k.preferences.set("note_guide", value)
    broadcast_event("preferences_update", {"key": "note_guide", "value": value})
    k.update_now_playing_socket()
    return jsonify({"success": True})


@notes_bp.route("/api/notes/retry", methods=["POST"])
@public
def retry():
    """Try again to make the playing song's chart, after it could not be made."""
    song = _playing()
    if song:
        get_karaoke_instance().chart_manager.retry(song)
    return jsonify({"success": bool(song)})


@notes_bp.route("/api/notes/status")
def status():
    """Whether charts can be made here, and what is being made."""
    k = get_karaoke_instance()
    return jsonify(
        {
            "available": k.chart_manager.available,
            "enabled": bool(k.preferences.get_or_default("make_charts")),
            **k.chart_manager.progress(),
        }
    )


@notes_bp.route("/api/notes/make_all", methods=["POST"])
def make_all():
    """Make charts for every song in the library that has none, in the background."""
    k = get_karaoke_instance()
    if not k.chart_manager.available:
        return jsonify({"success": False, "queued": 0}), 400
    return jsonify({"success": True, "queued": k.make_all_charts()})

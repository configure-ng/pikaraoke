"""Makes each song's note chart once, in the background, with the chart add-on.

A chart is made from the song's original recording, the one its lyrics were
lined up against, or failing that one found on YouTube by name and checked
against the video. The heavy work (separating the vocals, tracking their
pitch) runs in a child process, chart_maker.py, one song at a time.
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable

import numpy as np
from gevent import sleep, spawn

from pikaraoke.lib import chart
from pikaraoke.lib.events import EventSystem
from pikaraoke.lib.karaoke_database import KaraokeDatabase
from pikaraoke.lib.lyrics import Aligned, locate_original, next_in_line
from pikaraoke.lib.score_keeper import Melody
from pikaraoke.lib.youtube_dl import download_audio

# Charts need far better audio than lining up does: the voice is separated
# from it. Anything over a few minutes on the CPU is still a normal song.
_MAKE_TIMEOUT_S = 30 * 60
# YouTube now and then refuses a download (HTTP 403) that works moments later.
_DOWNLOAD_ATTEMPTS = 3
_DOWNLOAD_RETRY_WAIT_S = 5


def _make_pitch(audio_path: str, out_path: str, model_dir: str) -> bool:
    """Run chart_maker in its own process, at low priority; whether it succeeded."""
    cmd = [sys.executable, "-m", "pikaraoke.lib.chart_maker", audio_path, out_path]
    cmd += ["--model-dir", model_dir]
    kwargs: dict = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.BELOW_NORMAL_PRIORITY_CLASS
    else:
        kwargs["preexec_fn"] = lambda: os.nice(10)
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=_MAKE_TIMEOUT_S, **kwargs)
    except (subprocess.TimeoutExpired, OSError) as e:
        logging.warning(f"Chart: making failed: {e}")
        return False
    for line in result.stderr.decode("utf-8", "ignore").splitlines():
        if line.startswith("Chart:"):
            logging.info(line)
    if result.returncode != 0:
        logging.warning(f"Chart: making failed: {result.stderr.decode('utf-8', 'ignore')[-500:]}")
        return False
    return True


def _addon_available() -> bool:
    # Imported here, not at the top: chart_maker runs as its own process with
    # -m, and importing it with the package first makes runpy warn.
    from pikaraoke.lib.chart_maker import available

    return available()


def _download_original(
    url: str, dest_dir: str, wait: Callable[[float], None] = sleep
) -> str | None:
    """The original's audio, asking again if YouTube refuses at first."""
    for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
        path = download_audio(url, dest_dir, audio_format="bestaudio/best", timeout=300)
        if path is not None:
            return path
        if attempt < _DOWNLOAD_ATTEMPTS:
            logging.info(
                f"Chart: download failed, trying again ({attempt + 1}/{_DOWNLOAD_ATTEMPTS})"
            )
            wait(_DOWNLOAD_RETRY_WAIT_S)
    return None


def notes_from_file(path: str) -> list[chart.Note]:
    """The notes in a pitch curve chart_maker wrote."""
    with open(path) as f:
        data = json.load(f)
    midi = np.array([np.nan if m is None else m for m in data["midi"]], dtype=float)
    confidence = np.array(data["confidence"], dtype=float)
    times = np.arange(len(midi)) * data["hop"]
    return chart.notes_from_pitch(times, midi, confidence)


class ChartManager:
    """Makes note charts for songs, one at a time, and serves them."""

    def __init__(
        self,
        db: KaraokeDatabase,
        events: EventSystem,
        model_dir: str,
        get_now_playing: Callable[[], str | None],
        get_query: Callable[[str, dict], str],
        get_duration: Callable[[str], float | None],
        notify: Callable[[], None],
        is_enabled: Callable[[], bool] = lambda: True,
        available: Callable[[], bool] = lambda: _addon_available(),
        locate: Callable[..., Aligned | None] = locate_original,
        download: Callable[[str, str], str | None] = _download_original,
        make_pitch: Callable[[str, str, str], bool] = _make_pitch,
        spawn: Callable = spawn,
        queue_position: Callable[[str], int | None] = lambda song: None,
    ) -> None:
        self._db = db
        self._queue_position = queue_position
        # Failed songs retried by retry_failed this run, so each is tried once more.
        self._retried: set[str] = set()
        self._model_dir = model_dir
        self._get_now_playing = get_now_playing
        self._get_query = get_query
        self._get_duration = get_duration
        self._notify = notify
        self._is_enabled = is_enabled
        self._available = available
        self._locate = locate
        self._download = download
        self._make_pitch = make_pitch
        self._spawn = spawn
        self._pending: list[str] = []
        self._making: str | None = None
        self._working = False
        self._is_available: bool | None = None
        # After the lyrics lookup, which may have found the original already.
        events.on("lyrics_looked_up", self.request)
        events.on("playback_started", self._on_playback_started)

    def _on_playback_started(self) -> None:
        song = self._get_now_playing()
        lyrics = self._db.get_lyrics(song) if song else None
        # A song not yet looked up gets its chart when the lookup ends.
        if lyrics is not None and lyrics["status"] is not None:
            self.request(song)

    @property
    def available(self) -> bool:
        """Whether the add-on is installed; checked once."""
        if self._is_available is None:
            self._is_available = self._available()
        return self._is_available

    def request(self, song_path: str) -> None:
        """Make a song's chart in the background, unless that has been tried."""
        if not self._is_enabled() or not self.available:
            return
        if song_path in self._pending or song_path == self._making:
            return
        row = self._db.get_chart(song_path)
        if row is None or row["status"] is not None:
            return
        self._pending.append(song_path)
        if not self._working:
            self._working = True
            self._spawn(self._work)

    def request_all(self) -> int:
        """Queue every song not yet tried; how many were queued."""
        before = len(self._pending)
        for path in self._db.get_paths_without_chart():
            self.request(path)
        return len(self._pending) - before

    def _work(self) -> None:
        try:
            while self._pending:
                # The song playing, then those queued to sing: someone may be waiting.
                song = self._making = next_in_line(
                    self._pending, self._get_now_playing(), self._queue_position
                )
                self._changed(song)
                try:
                    self._make(song)
                except Exception:  # one bad song must not stop the rest
                    logging.exception(f"Chart: making failed for {song}")
                    self._db.set_chart(song, None, "failed")
                finally:
                    self._making = None
                self._changed(song)
        finally:
            self._working = False

    def _make(self, song_path: str) -> None:
        row = self._db.get_chart(song_path)
        if row is None:
            return
        url = row["original_url"]
        if not url:
            found = self._locate(
                song_path, self._get_query(song_path, row), self._get_duration(song_path)
            )
            if found is None:
                logging.info(f"Chart: no original recording found for {song_path}")
                self._db.set_chart(song_path, None, "missing")
                return
            url = found.url
            self._db.set_original(song_path, found.url, found.offset)

        with tempfile.TemporaryDirectory(prefix="pikaraoke-chart-") as tmp:
            audio = self._download(url, tmp)
            if audio is None:
                self._db.set_chart(song_path, None, "failed")
                return
            out = os.path.join(tmp, "pitch.json")
            if not self._make_pitch(audio, out, self._model_dir):
                self._db.set_chart(song_path, None, "failed")
                return
            notes = notes_from_file(out)
        if not notes:
            self._db.set_chart(song_path, None, "missing")
            return
        self._db.set_chart(song_path, chart.to_json(notes, "generated", original=url), "found")
        logging.info(f"Chart: {len(notes)} notes for {os.path.basename(song_path)}")

    def _changed(self, song_path: str) -> None:
        if song_path == self._get_now_playing():
            self._notify()

    def status(self, song_path: str | None) -> str | None:
        """'making', 'found', 'missing', 'failed', or None if never tried."""
        if not song_path:
            return None
        if song_path == self._making or song_path in self._pending:
            return "making"
        row = self._db.get_chart(song_path)
        return row["status"] if row else None

    def progress(self) -> dict:
        """What the settings page shows of the work in hand."""
        return {"making": self._making, "queued": len(self._pending)}

    def melody(self, song_path: str | None) -> Melody | None:
        """A song's chart for scoring, or None if it has none."""
        row = self._db.get_chart(song_path) if song_path else None
        if not row or not row["chart"]:
            return None
        return Melody(chart.from_json(row["chart"]), row["offset"])

    def notes(self, song_path: str | None) -> list[list] | None:
        """A song's chart for the TV, as [start, length, pitch] rows in the
        recording's time (the TV adds the song's offset, as for lyrics)."""
        row = self._db.get_chart(song_path) if song_path else None
        if not row or not row["chart"]:
            return None
        return json.loads(row["chart"])["notes"]

    def retry(self, song_path: str) -> None:
        row = self._db.get_chart(song_path)
        if row is not None and row["status"] in ("missing", "failed"):
            self._db.forget_chart(song_path)
            self.request(song_path)

    def retry_failed(self, song_path: str) -> None:
        """Try a failed chart once more, at most once a run: a download YouTube
        refused, or a machine busy at the time, may well work now."""
        if song_path in self._retried:
            return
        row = self._db.get_chart(song_path)
        if row is not None and row["status"] == "failed":
            self._retried.add(song_path)
            self.retry(song_path)

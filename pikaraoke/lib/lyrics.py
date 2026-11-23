"""Timed lyrics for songs whose video shows none, with romaji and kana for Japanese.

Karaoke tracks have no vocals to transcribe, so lyrics come from LRCLIB, an
open database of lyrics timed line by line. Those timings follow the original
recording, and a karaoke video rarely starts where the recording does (title
cards, a longer intro). So the original is fetched from YouTube too and the two
are lined up by their harmony (the chroma the singing scores use): music in
the same key at the same moment has the same chords, vocals or not.

Lookups run once per song, at download or on first play, one at a time in the
background. The TV shows them only when someone turns them on, per song.
"""

import json
import logging
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import requests
from gevent import get_hub, spawn

from pikaraoke.lib import japanese, pitch
from pikaraoke.lib.events import EventSystem
from pikaraoke.lib.karaoke_database import KaraokeDatabase
from pikaraoke.lib.metadata_parser import normalize_for_comparison
from pikaraoke.lib.score_keeper import decode_audio
from pikaraoke.lib.youtube_dl import download_audio, get_search_results

LRCLIB_SEARCH_URL = "https://lrclib.net/api/search"
# LRCLIB asks clients to name themselves; nothing about the user goes with it.
_USER_AGENT = "PiKaraoke (https://github.com/configure-ng/pikaraoke)"
_TIMEOUT_S = 15

JAPANESE_MODES = ["off", "romaji", "kana"]
OTHER_MODES = ["off", "on"]

_LRC_TAG = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]")
# A lyrics entry for a karaoke or instrumental cut carries no words to sing.
_NOT_THE_SONG = re.compile(
    r"inst|instrumental|karaoke|off\s*vocal|カラオケ|ニコカラ|オフボーカル", re.IGNORECASE
)

# How far apart the karaoke video and the original may start, either way.
_MAX_OFFSET_S = 60.0
# The original found on YouTube must be within this of the lyrics' duration,
# or it is a different cut and its timings would not carry over.
_ORIGINAL_DURATION_SLACK_S = 4.0
# The best alignment must beat the best one elsewhere by this ratio to be
# trusted; measured peaks run 2-3x, a song against a different song ~1.1x.
_MIN_PEAK_RATIO = 1.4
# Search results tried against the video when looking for an original by name.
_MAX_ORIGINALS_TRIED = 3
# Lags this close to the best (in chroma rows) are the same peak, not a rival.
_PEAK_WIDTH_ROWS = 20


@dataclass
class Found:
    """What a lookup found: the lyrics row JSON, how far to shift it, and the
    YouTube URL of the original recording it was lined up against, if any."""

    lyrics: str
    offset: float
    original: str | None = None


@dataclass
class Aligned:
    """An original recording found on YouTube and where it sits against the video."""

    url: str
    offset: float


def parse_lrc(lrc: str) -> list[tuple[float, str]]:
    """(start seconds, text) per line, in time order; an empty text is a break.

    A line with several time tags repeats at each. Where a time has two lines
    (a translation under the original), the first is kept.
    """
    lines: dict[float, str] = {}
    for raw in lrc.splitlines():
        tags = list(_LRC_TAG.finditer(raw))
        if not tags:
            continue
        text = raw[tags[-1].end() :].strip()
        for tag in tags:
            start = round(int(tag.group(1)) * 60 + float(tag.group(2)), 2)
            lines.setdefault(start, text)
    return sorted(lines.items())


def build_lyrics(lines: list[tuple[float, str]]) -> dict:
    """The stored form: each line with its romaji and kana when the song is Japanese."""
    is_japanese = any(japanese.is_japanese(text) for _, text in lines)
    out = []
    for start, text in lines:
        line = {"t": start, "text": text}
        if is_japanese:
            line["romaji"] = japanese.to_romaji(text) if text else ""
            line["kana"] = japanese.to_kana(text) if text else ""
        out.append(line)
    return {"japanese": is_japanese, "lines": out}


def search_lrclib(query: str) -> list[dict]:
    """LRCLIB's matches for a query, with timed lyrics only."""
    try:
        response = requests.get(
            LRCLIB_SEARCH_URL,
            params={"q": query},
            headers={"User-Agent": _USER_AGENT},
            timeout=_TIMEOUT_S,
        )
        response.raise_for_status()
        results = response.json()
    except (requests.RequestException, ValueError) as e:
        logging.warning(f"Lyrics: search for {query!r} failed: {e}")
        return []
    return [r for r in results if r.get("syncedLyrics") and not r.get("instrumental")]


def pick_entry(results: list[dict], query: str, duration: float | None) -> dict | None:
    """The entry most likely to be this song.

    It must share its title with the query; one sharing the artist too wins,
    then the one nearest the video's length, then LRCLIB's own order.
    """
    query_norm = normalize_for_comparison(query)
    parts = [p for p in re.split(r"\s+-\s+|\s*/\s*|\s+", query_norm) if len(p) >= 2]

    def matches(field: str) -> bool:
        norm = normalize_for_comparison(field)
        return bool(norm) and (norm in query_norm or any(p in norm for p in parts))

    candidates = [
        (not matches(r.get("artistName", "")), _length_gap(r, duration), i, r)
        for i, r in enumerate(results)
        if matches(r.get("trackName", "")) and not _NOT_THE_SONG.search(r.get("trackName", ""))
    ]
    return min(candidates, key=lambda c: c[:3])[3] if candidates else None


def _length_gap(entry: dict, duration: float | None) -> float:
    if not duration or not entry.get("duration"):
        return 0.0
    return abs(entry["duration"] - duration)


def alignment_offset(karaoke: np.ndarray, original: np.ndarray) -> float | None:
    """Seconds the karaoke track runs behind the original, or None if they do not match.

    Both are chroma from pitch.backing_chroma. Every transposition is tried,
    since karaoke tracks are sometimes in a different key from the record.
    """
    a, b = _comparable(karaoke), _comparable(original)
    if len(a) == 0 or len(b) == 0:
        return None
    max_lag = int(_MAX_OFFSET_S / pitch.CHROMA_HOP_S)
    best_scores, best_lag = None, 0
    for shift in range(12):
        lags, scores = _cross_correlate(a, np.roll(b, shift, axis=1), max_lag)
        if best_scores is None or scores.max() > best_scores.max():
            best_scores, best_lag = scores, lags[np.argmax(scores)]
    elsewhere = best_scores[np.abs(lags - best_lag) > _PEAK_WIDTH_ROWS]
    peak = best_scores.max()
    if peak <= 0 or (len(elsewhere) and peak < _MIN_PEAK_RATIO * max(elsewhere.max(), 1e-9)):
        return None
    return round(float(best_lag * pitch.CHROMA_HOP_S), 2)


def _comparable(chroma: np.ndarray) -> np.ndarray:
    """Chroma rows as unit-length contrast vectors, centred so silence scores nothing."""
    standout = np.clip(chroma - np.median(chroma, axis=1, keepdims=True), 0, None)
    unit = standout / (np.linalg.norm(standout, axis=1, keepdims=True) + 1e-9)
    return unit - unit.mean(axis=0)


def _cross_correlate(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean similarity of a[i + lag] to b[i] for each lag in +-max_lag."""
    size = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    spectrum = np.fft.rfft(a, size, axis=0) * np.conj(np.fft.rfft(b, size, axis=0))
    full = np.fft.irfft(spectrum.sum(axis=1), size)
    lags = np.arange(-max_lag, max_lag + 1)
    overlap = np.minimum(len(a) - lags, len(b)) - np.maximum(0, -lags)
    return lags, full[lags % size] / np.maximum(overlap, 1)


def find_original(entry: dict) -> list[str]:
    """YouTube URLs that may be the recording the lyrics were timed against, best first."""
    query = f"{entry.get('artistName', '')} {entry.get('trackName', '')}".strip()
    try:
        results = get_search_results(query)
    except Exception as e:  # yt-dlp failing must not fail the lookup
        logging.warning(f"Lyrics: no original found for {query!r}: {e}")
        return []
    urls = []
    for result in results:
        seconds = _seconds(result.duration)
        if seconds is None or _NOT_THE_SONG.search(result.title):
            continue
        if abs(seconds - entry.get("duration", 0)) <= _ORIGINAL_DURATION_SLACK_S:
            urls.append(result.url)
    return urls[:_MAX_ORIGINALS_TRIED]


def _seconds(duration: str) -> int | None:
    """'4:22' to 262."""
    minutes, _, seconds = duration.partition(":")
    return int(minutes) * 60 + int(seconds) if minutes.isdigit() and seconds.isdigit() else None


def measure_offset(
    song_path: str, entry: dict, run: Callable = lambda f, *args: f(*args)
) -> Aligned | None:
    """Line the lyrics up with this video: the original they follow, and the
    seconds to add to their timings."""
    # The first match can be age-restricted or region-locked; try the next.
    for url in find_original(entry):
        offset = align_to(song_path, url, run)
        if offset is not None:
            return Aligned(url, offset)
    return None


def align_to(song_path: str, url: str, run: Callable = lambda f, *args: f(*args)) -> float | None:
    """Seconds this video runs behind the recording at url, or None if they do not match."""
    with tempfile.TemporaryDirectory(prefix="pikaraoke-lyrics-") as tmp:
        original_path = download_audio(url, tmp)
        if original_path is None:
            return None
        original, karaoke = decode_audio(original_path), decode_audio(song_path)
        if original is None or karaoke is None:
            return None
        return run(
            lambda: alignment_offset(pitch.backing_chroma(karaoke), pitch.backing_chroma(original))
        )


def locate_original(
    song_path: str,
    query: str,
    duration: float | None,
    run: Callable = lambda f, *args: f(*args),
) -> Aligned | None:
    """Find the original recording of a song by searching YouTube for it.

    For songs with no lyrics entry to say which recording to look for: each
    likely result is tried against the video, and the first whose music lines
    up with it is taken. A cover or another song does not line up.
    """
    try:
        results = get_search_results(query)
    except Exception as e:  # yt-dlp failing must not fail the lookup
        logging.warning(f"Original: search for {query!r} failed: {e}")
        return None
    tried = 0
    for result in results:
        seconds = _seconds(result.duration)
        if seconds is None or _NOT_THE_SONG.search(result.title):
            continue
        if duration and abs(seconds - duration) > _MAX_OFFSET_S:
            continue
        offset = align_to(song_path, result.url, run)
        if offset is not None:
            logging.info(f"Original: {result.title!r} lines up, {offset:+.1f}s against the video")
            return Aligned(result.url, offset)
        tried += 1
        if tried >= _MAX_ORIGINALS_TRIED:
            break
    return None


def find_lyrics(
    song_path: str, query: str, duration: float | None, run: Callable = lambda f, *args: f(*args)
) -> Found | None:
    """Look a song up and line its lyrics up with the video; None if not found."""
    entry = pick_entry(search_lrclib(query), query, duration)
    if entry is None:
        logging.info(f"Lyrics: none found for {query!r}")
        return None
    lines = parse_lrc(entry["syncedLyrics"])
    if not any(text for _, text in lines):
        return None
    lyrics = run(build_lyrics, lines)
    lyrics["source"] = {"lrclib_id": entry.get("id"), "duration": entry.get("duration")}
    aligned = measure_offset(song_path, entry, run)
    lyrics["aligned"] = aligned is not None
    logging.info(
        f"Lyrics: found {entry.get('trackName')!r} by {entry.get('artistName')!r}"
        + (f", {aligned.offset:+.1f}s against the video" if aligned else ", unaligned")
    )
    return Found(
        lyrics=json.dumps(lyrics, ensure_ascii=False),
        offset=aligned.offset if aligned else 0.0,
        original=aligned.url if aligned else None,
    )


def _in_thread(f: Callable, *args):
    """Run CPU-bound work off the gevent loop, so the server keeps answering."""
    return get_hub().threadpool.apply(f, args)


def next_in_line(
    pending: list[str], now_playing: str | None, queue_position: Callable[[str], int | None]
) -> str:
    """Take the next song to work on from pending: the one playing, then songs
    in the singing queue in their order, then the rest as they came."""
    if now_playing in pending:
        pending.remove(now_playing)
        return now_playing
    queued = [
        (pos, i) for i, song in enumerate(pending) if (pos := queue_position(song)) is not None
    ]
    return pending.pop(min(queued)[1] if queued else 0)


class LyricsManager:
    """Looks up each song's lyrics once and keeps how the room wants them shown."""

    def __init__(
        self,
        db: KaraokeDatabase,
        events: EventSystem,
        get_now_playing: Callable[[], str | None],
        get_query: Callable[[str, dict], str],
        get_duration: Callable[[str], float | None],
        notify: Callable[[], None],
        is_enabled: Callable[[], bool] = lambda: True,
        find: Callable[..., Found | None] = find_lyrics,
        spawn: Callable = spawn,
        run: Callable = _in_thread,
        queue_position: Callable[[str], int | None] = lambda song: None,
    ) -> None:
        self._db = db
        self._queue_position = queue_position
        self._events = events
        self._get_now_playing = get_now_playing
        self._get_query = get_query
        self._get_duration = get_duration
        self._notify = notify
        self._is_enabled = is_enabled
        self._find = find
        self._spawn = spawn
        self._run = run
        self._pending: list[str] = []
        self._searching: str | None = None
        self._working = False
        events.on("playback_started", self._on_playback_started)

    def _on_playback_started(self) -> None:
        song = self._get_now_playing()
        if song:
            self.request(song)

    def request(self, song_path: str) -> None:
        """Look a song's lyrics up in the background, unless that has been done."""
        if not self._is_enabled() or song_path in self._pending or song_path == self._searching:
            return
        row = self._db.get_lyrics(song_path)
        if row is None or row["status"] is not None:
            return
        self._pending.append(song_path)
        if not self._working:
            self._working = True
            self._spawn(self._work)

    def _work(self) -> None:
        try:
            while self._pending:
                # The song playing, then those queued to sing: someone may be waiting.
                song = self._searching = next_in_line(
                    self._pending, self._get_now_playing(), self._queue_position
                )
                self._changed(song)
                try:
                    self._look_up(song)
                except Exception:  # one bad song must not stop the rest
                    logging.exception(f"Lyrics: lookup failed for {song}")
                finally:
                    self._searching = None
                self._changed(song)
                # The note chart is made from the original the lookup found.
                self._events.emit("lyrics_looked_up", song)
        finally:
            self._working = False

    def _look_up(self, song_path: str) -> None:
        row = self._db.get_lyrics(song_path)
        if row is None:
            return
        found = self._find(
            song_path, self._get_query(song_path, row), self._get_duration(song_path), self._run
        )
        if found is None:
            self._db.set_lyrics(song_path, None)
        else:
            self._db.set_lyrics(song_path, found.lyrics, found.offset)
            if found.original:
                self._db.set_original(song_path, found.original, found.offset)

    def _changed(self, song_path: str) -> None:
        if song_path == self._get_now_playing():
            self._notify()

    def state(self, song_path: str | None) -> dict | None:
        """What the phone needs to show the lyrics controls for a song."""
        if not song_path:
            return None
        row = self._db.get_lyrics(song_path)
        if row is None:
            return None
        if song_path == self._searching or song_path in self._pending:
            status = "searching"
        else:
            status = row["status"]
        modes = OTHER_MODES
        if row["lyrics"]:
            modes = JAPANESE_MODES if json.loads(row["lyrics"]).get("japanese") else OTHER_MODES
        return {
            "status": status,
            "modes": modes,
            "mode": row["mode"] if row["mode"] in modes else "off",
            "offset": row["offset"],
        }

    def lines(self, song_path: str | None) -> dict | None:
        """A song's lyrics for the TV to show, or None if it has none."""
        row = self._db.get_lyrics(song_path) if song_path else None
        if not row or not row["lyrics"]:
            return None
        lyrics = json.loads(row["lyrics"])
        return {"japanese": lyrics.get("japanese", False), "lines": lyrics["lines"]}

    def set_mode(self, song_path: str, mode: str) -> bool:
        state = self.state(song_path)
        if state is None or mode not in state["modes"]:
            return False
        self._db.set_lyrics_display(song_path, mode, state["offset"])
        self._notify()
        return True

    def nudge(self, song_path: str, seconds: float) -> bool:
        """Move the lyrics later (+) or earlier (-) against the video."""
        state = self.state(song_path)
        if state is None:
            return False
        offset = round(max(-_MAX_OFFSET_S, min(_MAX_OFFSET_S, state["offset"] + seconds)), 2)
        self._db.set_lyrics_display(song_path, state["mode"], offset)
        self._notify()
        return True

    def retry(self, song_path: str) -> None:
        """Look again for a song where nothing was found."""
        row = self._db.get_lyrics(song_path)
        if row is not None and row["status"] == "missing":
            self._db.forget_lyrics(song_path)
            self.request(song_path)

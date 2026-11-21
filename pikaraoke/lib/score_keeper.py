"""Scores singing from the mics against the song that is playing.

Each song's backing track is analysed when it starts (see pitch.py for how),
mic audio is pitch-tracked while it plays, and the score screen asks for the
result when it ends. Nothing is scored without a mic, so the screen is skipped.
"""

import logging
import os
import subprocess
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from gevent import spawn

from pikaraoke.lib import pitch
from pikaraoke.lib.events import EventSystem

_FRAME_HOP_S = 0.02
# Each frame's harmony is the backing averaged over this many chroma rows
# either side, smoothing over single-frame drum hits and note onsets.
_HARMONY_CONTEXT_ROWS = 3

# Louder than this (full scale = 1.0), else it is room noise or the speakers
# leaking into a mic left on the table; and louder than the mic's own quiet
# moments by _GATE_OVER_FLOOR, so steady bleed never counts as singing.
_MIN_LEVEL = 0.008
_GATE_OVER_FLOOR = 4.0
_FLOOR_PERCENTILE = 10

# Songs are mostly sung, with breaths, gaps and solos: a singer voicing this
# share of the song has sung all of it.
_FULL_PARTICIPATION = 0.35
# A mic must carry this much singing to be scored as a singer of its own,
# so a second mic left on but unused does not drag a duet down.
_MIN_SUNG_S = 3.0
# Mean harmony fit mapped to 0-1 accuracy. Calibrated on synthetic singing:
# random pitches average ~0.2, a melody a semitone off ~0.1, and the melody
# sung in tune, with vibrato and drift, ~0.55.
_FIT_OF_NO_SKILL = 0.15
_FIT_OF_FULL_SKILL = 0.5
# Share of the score that showing up and singing earns, accuracy the rest.
_EFFORT_SHARE = 0.15

# Ahead of the last position report, playback is assumed to run on at most
# this far; reports come each second, so a stall shows within a beat.
_MAX_EXTRAPOLATION_S = 1.5


@dataclass
class PlaybackState:
    playback_id: str | None
    file_path: str | None
    semitones: int
    position: float | None
    is_paused: bool


@dataclass
class _MicTrack:
    """One mic's pitch frames for the current song."""

    samplerate: int
    pending: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    positions: list[float] = field(default_factory=list)
    levels: list[float] = field(default_factory=list)
    notes: list[float | None] = field(default_factory=list)


@dataclass
class _Backing:
    chroma: np.ndarray
    song_profile: np.ndarray


class ScoreKeeper:
    """Turns mic audio during a song into a score for each singer."""

    def __init__(
        self,
        events: EventSystem,
        set_capture: Callable[[Callable[[str, bytes, int], None] | None], None],
        get_playback: Callable[[], PlaybackState],
        is_enabled: Callable[[], bool] = lambda: True,
        spawn: Callable = spawn,
    ) -> None:
        self._set_capture = set_capture
        self._get_playback = get_playback
        # Checked per song, so the score setting applies from the next one.
        self._is_enabled = is_enabled
        self._spawn = spawn
        self._playback_id: str | None = None
        self._backing: _Backing | None = None
        self._mics: dict[str, _MicTrack] = {}
        self._anchor: tuple[float, float] | None = None
        events.on("playback_started", self._on_playback_started)
        events.on("song_ended", lambda *_: self._set_capture(None))

    def _on_playback_started(self) -> None:
        playback = self._get_playback()
        self._playback_id = playback.playback_id
        self._backing = None
        self._mics = {}
        self._anchor = None
        if not self._is_enabled():
            return
        if playback.file_path:
            self._spawn(
                self._analyse_backing,
                playback.playback_id,
                playback.file_path,
                playback.semitones,
            )
        self._set_capture(self.feed)

    def _analyse_backing(self, playback_id: str | None, file_path: str, semitones: int) -> None:
        pcm = _decode_audio(file_path)
        if pcm is None:
            return
        chroma = pitch.backing_chroma(pcm)
        # The stream plays transposed, so the singer hears the music shifted.
        chroma = np.roll(chroma, semitones, axis=1)
        if playback_id != self._playback_id or len(chroma) == 0:
            return
        self._backing = _Backing(chroma=chroma, song_profile=chroma.mean(axis=0))
        logging.info(f"Score: analysed backing of {os.path.basename(file_path)}")

    def feed(self, mic: str, pcm: bytes, samplerate: int) -> None:
        """Take a chunk of a mic's audio (mono int16) and pitch-track it."""
        position = self._position()
        track = self._mics.setdefault(mic, _MicTrack(samplerate=samplerate))
        if position is None:
            # Paused or not yet playing: the audio belongs to no moment of the song.
            track.pending = track.pending[:0]
            return

        frame_len = pitch.pitch_frame_length(samplerate)
        hop = int(samplerate * _FRAME_HOP_S)
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768
        buffered = np.concatenate((track.pending, samples))
        # Frames end at the newest sample, which is at `position`.
        n_frames = max(0, 1 + (len(buffered) - frame_len) // hop)
        for i in range(n_frames):
            frame = buffered[i * hop : i * hop + frame_len]
            track.positions.append(position - (len(buffered) - i * hop - frame_len) / samplerate)
            track.levels.append(float(np.sqrt(np.mean(frame * frame))))
            frequency = pitch.detect_pitch(frame, samplerate)
            track.notes.append(None if frequency is None else pitch.midi_note(frequency))
        track.pending = buffered[n_frames * hop :]

    def _position(self) -> float | None:
        """Current song position, interpolated between the once-a-second reports."""
        playback = self._get_playback()
        if playback.is_paused or playback.position is None:
            self._anchor = None
            return None
        now = time.monotonic()
        if self._anchor is None or self._anchor[0] != playback.position:
            self._anchor = (playback.position, now)
        return playback.position + min(now - self._anchor[1], _MAX_EXTRAPOLATION_S)

    def result(self, playback_id: str | None) -> dict | None:
        """Scores for the song with this playback id, or None if there is nothing to score.

        One {"number", "score"} per singing mic, numbered in mic-name order so a
        mic keeps its number whoever sings, and their average when there are
        several. Mics on but silent show only if nobody sang.
        """
        if playback_id is None or playback_id != self._playback_id:
            return None
        if self._backing is None or not self._mics:
            return None

        scored = [
            (number, mic, *self._score(track, self._backing))
            for number, (mic, track) in enumerate(sorted(self._mics.items()), start=1)
        ]
        singers = [entry for entry in scored if entry[3] >= _MIN_SUNG_S]
        shown = singers or scored
        logging.info(f"Score: {', '.join(f'{mic}={score}' for _, mic, score, _ in shown)}")
        scores = [score for _, _, score, _ in shown]
        return {
            "mics": [{"number": number, "score": score} for number, _, score, _ in shown],
            "combined": round(sum(scores) / len(scores)) if len(scores) > 1 else None,
        }

    def _score(self, track: _MicTrack, backing: _Backing) -> tuple[int, float]:
        """A mic's score from 0 to 99, and how many seconds of singing it carried."""
        if not track.levels:
            return 0, 0.0
        levels = np.array(track.levels)
        gate = max(_MIN_LEVEL, _GATE_OVER_FLOOR * np.percentile(levels, _FLOOR_PERCENTILE))

        fits = []
        for position, level, note in zip(track.positions, levels, track.notes):
            if note is None or level < gate:
                continue
            row = int(position / pitch.CHROMA_HOP_S)
            if not 0 <= row < len(backing.chroma):
                continue
            local = backing.chroma[
                max(0, row - _HARMONY_CONTEXT_ROWS) : row + _HARMONY_CONTEXT_ROWS + 1
            ].mean(axis=0)
            fits.append(pitch.harmony_fit(note, pitch.harmony_profile(local, backing.song_profile)))

        if not fits:
            return 0, 0.0
        sung_s = len(fits) * _FRAME_HOP_S
        listened_s = max(track.positions) - min(track.positions)
        participation = min(1.0, sung_s / max(_FULL_PARTICIPATION * listened_s, _MIN_SUNG_S))
        accuracy = np.clip(
            (np.mean(fits) - _FIT_OF_NO_SKILL) / (_FIT_OF_FULL_SKILL - _FIT_OF_NO_SKILL), 0, 1
        )
        score = 99 * (_EFFORT_SHARE + (1 - _EFFORT_SHARE) * accuracy) * participation
        return int(round(score)), sung_s


def _decode_audio(file_path: str) -> np.ndarray | None:
    """A song's audio as mono float samples at pitch.BACKING_SAMPLERATE.

    A zipped CDG's audio is the mp3 inside the archive, streamed to ffmpeg.
    """
    source, stdin_data = file_path, None
    if os.path.splitext(file_path)[1].casefold() == ".zip":
        try:
            with zipfile.ZipFile(file_path) as archive:
                mp3 = next(n for n in archive.namelist() if n.casefold().endswith(".mp3"))
                stdin_data = archive.read(mp3)
        except (OSError, zipfile.BadZipFile, StopIteration) as e:
            logging.warning(f"Score: no audio in {file_path}: {e}")
            return None
        source = "pipe:0"

    # -nostdin keeps ffmpeg off the server's stdin, except where stdin is the input.
    cmd = ["ffmpeg"] + (["-nostdin"] if stdin_data is None else [])
    cmd += ["-v", "error", "-i", source, "-vn", "-ac", "1"]
    cmd += ["-ar", str(pitch.BACKING_SAMPLERATE), "-f", "s16le", "-"]
    try:
        result = subprocess.run(cmd, input=stdin_data, capture_output=True, timeout=120)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logging.warning(f"Score: could not decode {file_path}: {e}")
        return None
    if result.returncode != 0:
        logging.warning(f"Score: ffmpeg failed on {file_path}: {result.stderr.decode()[-300:]}")
        return None
    return np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768

"""Scores singing from the mics against the song that is playing.

Mic audio is pitch-tracked while a song plays, streamed to the TV as it comes
(for the note guide), and scored when the song ends. A song with a note chart
is scored by the notes of its melody hit; one without, by how well the singing
fits the backing track's harmony, analysed when the song starts (see pitch.py).
Nothing is scored without a mic, so the score screen is skipped.
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
from pikaraoke.lib.chart import MelodyIndex, Note, fold_octave
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
_GATE_UNDER_LOUD = 0.25
_LOUD_PERCENTILE = 95

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

# A chart note counts as sung in full when this share of it carries singing on
# the note: consonants, breaths and onsets take the rest.
_FULL_NOTE_SHARE = 0.6
# Singing within _NOTE_SLACK of a chart note, in semitones, is on it; beyond,
# credit falls away with _NOTE_TOLERANCE, to a tenth at a semitone off.
_NOTE_SLACK = 0.25
_NOTE_TOLERANCE = 0.35
# Mean note score mapped to 0-1 accuracy, as for harmony: calibrated on
# synthetic singing, the melody sung with vibrato and drift scores ~0.8, the
# melody a semitone off ~0.25 and random notes ~0.1.
_NOTES_OF_NO_SKILL = 0.15
_NOTES_OF_FULL_SKILL = 0.7

# Live pitch for the TV: a mic's recent loudness levels, from which its quiet
# floor is judged as for scoring.
_LIVE_LEVELS = 250

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


@dataclass
class Melody:
    """A song's chart notes, in its recording's time, and the offset onto the video."""

    notes: list[Note]
    offset: float


class ScoreKeeper:
    """Turns mic audio during a song into a score for each singer."""

    def __init__(
        self,
        events: EventSystem,
        set_capture: Callable[[Callable[[str, bytes, int], None] | None], None],
        get_playback: Callable[[], PlaybackState],
        is_enabled: Callable[[], bool] = lambda: True,
        spawn: Callable = spawn,
        get_melody: Callable[[str], Melody | None] = lambda path: None,
        on_pitch: Callable[[dict], None] | None = None,
    ) -> None:
        self._set_capture = set_capture
        self._get_playback = get_playback
        # Checked per song, so the score setting applies from the next one.
        self._is_enabled = is_enabled
        self._spawn = spawn
        self._get_melody = get_melody
        self._on_pitch = on_pitch
        self._playback_id: str | None = None
        self._backing: _Backing | None = None
        self._melody: MelodyIndex | None = None
        self._mics: dict[str, _MicTrack] = {}
        self._anchor: tuple[float, float] | None = None
        events.on("playback_started", self._on_playback_started)
        events.on("song_ended", lambda *_: self._set_capture(None))

    def _on_playback_started(self) -> None:
        playback = self._get_playback()
        self._playback_id = playback.playback_id
        self._backing = None
        self._melody = None
        self._mics = {}
        self._anchor = None
        if not self._is_enabled():
            return
        melody = self._get_melody(playback.file_path) if playback.file_path else None
        if melody and melody.notes:
            self._melody = MelodyIndex(melody.notes, melody.offset, playback.semitones)
        elif playback.file_path:
            self._spawn(
                self._analyse_backing,
                playback.playback_id,
                playback.file_path,
                playback.semitones,
            )
        self._set_capture(self.feed)

    def _analyse_backing(self, playback_id: str | None, file_path: str, semitones: int) -> None:
        pcm = decode_audio(file_path)
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
        first = len(track.notes)
        for i in range(n_frames):
            frame = buffered[i * hop : i * hop + frame_len]
            track.positions.append(position - (len(buffered) - i * hop - frame_len) / samplerate)
            track.levels.append(float(np.sqrt(np.mean(frame * frame))))
            frequency = pitch.detect_pitch(frame, samplerate)
            track.notes.append(None if frequency is None else pitch.midi_note(frequency))
        track.pending = buffered[n_frames * hop :]
        if self._on_pitch is not None and n_frames:
            self._send_pitch(mic, track, first)

    def _send_pitch(self, mic: str, track: _MicTrack, first: int) -> None:
        """Tell the TV what a mic just sang: [song position, note or None] per frame."""
        gate = _gate(np.array(track.levels[-_LIVE_LEVELS:]))
        frames = [
            [
                round(position, 3),
                round(float(note), 2) if note is not None and level >= gate else None,
            ]
            for position, level, note in zip(
                track.positions[first:], track.levels[first:], track.notes[first:]
            )
        ]
        self._on_pitch({"mic": self.mic_number(mic), "frames": frames})

    def mic_number(self, mic: str) -> int:
        """Mics are numbered in name order, as on the score screen."""
        return sorted(self._mics).index(mic) + 1

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
        if (self._backing is None and self._melody is None) or not self._mics:
            return None

        scored = [
            (number, mic, *self._score(track))
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

    def _score(self, track: _MicTrack) -> tuple[int, float]:
        """A mic's score from 0 to 99, and how many seconds of singing it carried."""
        if not track.levels:
            return 0, 0.0
        if self._melody is not None:
            return self._score_notes(track, self._melody)
        return self._score_harmony(track, self._backing)

    def _score_notes(self, track: _MicTrack, melody: MelodyIndex) -> tuple[int, float]:
        """Scored by the chart: how much of each note was sung on it, octave aside."""
        levels = np.array(track.levels)
        gate = _gate(levels)
        sung = np.zeros(len(melody))
        sung_s = 0.0
        for position, level, note in zip(track.positions, levels, track.notes):
            if note is None or level < gate:
                continue
            sung_s += _FRAME_HOP_S
            i = melody.at(position)
            if i is None:
                continue
            target = melody.pitch(i)
            off = abs(fold_octave(note, target) - target)
            sung[i] += _FRAME_HOP_S * np.exp(
                -0.5 * (max(0.0, off - _NOTE_SLACK) / _NOTE_TOLERANCE) ** 2
            )

        if sung_s < _FRAME_HOP_S:
            return 0, 0.0
        # Only the notes in the stretch the mic listened to count.
        start, end = min(track.positions), max(track.positions)
        heard = [i for i in range(len(melody)) if start <= melody.start(i) < end]
        if not heard:
            return 0, sung_s
        lengths = np.array([melody.length(i) for i in heard])
        shares = np.minimum(1.0, sung[heard] / (_FULL_NOTE_SHARE * lengths))
        accuracy = np.clip(
            (np.average(shares, weights=lengths) - _NOTES_OF_NO_SKILL)
            / (_NOTES_OF_FULL_SKILL - _NOTES_OF_NO_SKILL),
            0,
            1,
        )
        score = 99 * (_EFFORT_SHARE + (1 - _EFFORT_SHARE) * accuracy)
        return int(round(score)), sung_s

    def _score_harmony(self, track: _MicTrack, backing: _Backing) -> tuple[int, float]:
        """Scored by the backing track: how well the notes sung fit its chords."""
        levels = np.array(track.levels)
        gate = _gate(levels)

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


def _gate(levels: np.ndarray) -> float:
    """The level a frame must reach to count as singing on this mic.

    Over the mic's quiet floor, but never over a share of its loud moments: a
    singer who has not paused yet has no quiet floor to measure.
    """
    if len(levels) == 0:
        return _MIN_LEVEL
    floor = _GATE_OVER_FLOOR * float(np.percentile(levels, _FLOOR_PERCENTILE))
    loud = _GATE_UNDER_LOUD * float(np.percentile(levels, _LOUD_PERCENTILE))
    return max(_MIN_LEVEL, min(floor, loud))


def decode_audio(file_path: str) -> np.ndarray | None:
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

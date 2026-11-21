"""Server-side microphone passthrough manager.

Enumerates system audio input devices and manages passthrough from
microphone inputs to speakers. Uses PulseAudio/PipeWire on Linux
(via pactl) and sounddevice (PortAudio) on Windows/macOS.

USB mics switch themselves on the first time they are seen, and on Windows
plugging or unplugging a device is picked up while running, so a karaoke mic
works without visiting the settings page.
"""

import ctypes
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from array import array
from collections import deque

from pikaraoke.lib.events import EventSystem
from pikaraoke.lib.get_platform import is_linux, is_windows
from pikaraoke.lib.preference_manager import PreferenceManager

_HAS_PACTL = is_linux() and shutil.which("pactl") is not None

# pactl needs XDG_RUNTIME_DIR to reach the per-user PulseAudio/PipeWire socket.
# It is unset when pikaraoke runs as a non-login service user (e.g. via systemd),
# so default it to the running user's runtime dir. Child pactl processes inherit
# os.environ, so setting it here is enough.
if _HAS_PACTL and not os.environ.get("XDG_RUNTIME_DIR"):
    # os.getuid is Unix-only; the _HAS_PACTL guard keeps this off Windows.
    os.environ["XDG_RUNTIME_DIR"] = f"/run/user/{os.getuid()}"  # pylint: disable=no-member
    logging.info(
        "XDG_RUNTIME_DIR was unset; defaulting to %s for pactl",
        os.environ["XDG_RUNTIME_DIR"],
    )

# sounddevice is only needed on non-Linux platforms
_SOUNDDEVICE_AVAILABLE = False
sd = None
if not _HAS_PACTL:
    try:
        import sounddevice as sd

        _SOUNDDEVICE_AVAILABLE = True
    except (ImportError, OSError):
        pass

# Above 1.0 boosts quiet mics (USB game mics are often quiet), clipping at full scale.
_MAX_GAIN = 2.0
_DEFAULT_LATENCY_MS = 10
_MIN_LATENCY_MS = 10
_MAX_LATENCY_MS = 200

# Size of each chunk handed from a mic's input stream to its output stream.
_BLOCK_MS = 5
# Chunks the input may run ahead of the output, beyond the latency setting,
# before the oldest is dropped. Absorbs the bursty delivery of audio drivers.
_DRIFT_SLACK_BLOCKS = 4

_HOTPLUG_POLL_SECONDS = 2.0

# Windows virtual/alias device names that duplicate real devices
_VIRTUAL_INPUT_DEVICE_NAMES = {
    "Microsoft Sound Mapper - Input",
    "Primary Sound Capture Driver",
}


def _pactl_list_sources() -> list[dict]:
    """List PulseAudio/PipeWire input sources with descriptions via a single pactl call."""
    try:
        result = subprocess.run(
            ["pactl", "list", "sources"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return []
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []

    sources = []
    current: dict[str, str] = {}
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("Source #"):
            if current.get("name") and ".monitor" not in current["name"]:
                sources.append(current)
            current = {"deviceId": stripped.split("#", 1)[1]}
        name_match = re.match(r"Name:\s+(.+)", stripped)
        if name_match:
            current["name"] = name_match.group(1)
        desc_match = re.match(r"Description:\s+(.+)", stripped)
        if desc_match:
            current["description"] = desc_match.group(1)

    # Don't forget the last source
    if current.get("name") and ".monitor" not in current["name"]:
        sources.append(current)

    return [
        {
            "deviceId": s["deviceId"],
            "label": s.get("description", s["name"]),
            "paSource": s["name"],
        }
        for s in sources
    ]


class _ActiveMic:
    """State for a single active microphone."""

    def __init__(
        self,
        device_id: str,
        loopback_module_id: str | None,
        streams: list | None = None,
        passthrough: "_Passthrough | None" = None,
        echo_module_id: str | None = None,
    ) -> None:
        self.device_id = device_id
        self.loopback_module_id = loopback_module_id
        self.streams = streams or []
        self.passthrough = passthrough
        self.echo_module_id = echo_module_id


class _Passthrough:
    """Carries one mic's audio from its input stream to an output stream.

    Two streams rather than one duplex stream: a USB mic and the PC's speakers
    run on separate clocks, often at different rates, which duplex cannot span.
    The bounded queue absorbs the drift between them: when the mic runs ahead
    the oldest audio is dropped, when it falls behind the output plays silence.

    Both callbacks run on PortAudio's own threads, so they touch only the
    deque (atomic appends and pops) and never log or take locks.
    """

    def __init__(self, gain: float, out_channels: int, max_queued_blocks: int) -> None:
        self.gain = gain
        self._out_channels = out_channels
        self._queue: deque[bytes] = deque(maxlen=max_queued_blocks)
        self._pending = bytearray()

    def on_input(self, indata, frames, time_info, status) -> None:
        self._queue.append(_mix_block(bytes(indata), self.gain, self._out_channels))

    def on_output(self, outdata, frames, time_info, status) -> None:
        needed = len(outdata)
        while len(self._pending) < needed and self._queue:
            self._pending += self._queue.popleft()
        chunk = bytes(self._pending[:needed])
        del self._pending[:needed]
        outdata[:] = chunk + b"\x00" * (needed - len(chunk))


class SoundManager:
    """Manages microphone enumeration and audio passthrough.

    On Linux (with PulseAudio or PipeWire), uses pactl module-loopback for
    zero-code audio routing. On Windows/macOS, uses sounddevice (PortAudio),
    preferring WASAPI on Windows for its far lower latency than MME.
    """

    def __init__(
        self, preferences: PreferenceManager, events: EventSystem, enabled: bool = False
    ) -> None:
        self._preferences = preferences
        self._events = events
        self._enabled = enabled
        self._active_mics: dict[str, _ActiveMic] = {}
        self._device_list: list[dict] = []
        self._device_index: dict[str, dict] = {}
        self._watching = False

        if enabled and not self.available:
            logging.warning(
                "sounddevice not available (missing PortAudio?). Microphone support disabled."
            )

    @property
    def available(self) -> bool:
        return self._enabled and (_HAS_PACTL or _SOUNDDEVICE_AVAILABLE)

    def enumerate_devices(self) -> list[dict]:
        """Query system audio input devices."""
        if not self.available:
            return []
        if _HAS_PACTL:
            return self._enumerate_pactl()
        if _SOUNDDEVICE_AVAILABLE:
            return self._enumerate_sounddevice()
        return []

    def _enumerate_pactl(self) -> list[dict]:
        """Enumerate mic devices via pactl."""
        self._device_list = _pactl_list_sources()
        self._device_index = {d["deviceId"]: d for d in self._device_list}
        logging.info(f"Mic devices enumerated (pactl): {[m['label'] for m in self._device_list]}")
        return self._device_list

    def _enumerate_sounddevice(self) -> list[dict]:
        """Enumerate mic devices via sounddevice (Windows/macOS)."""
        if not self._active_mics:
            _reinit_portaudio()

        try:
            devices = sd.query_devices()
        except sd.PortAudioError as e:
            logging.error(f"Failed to query audio devices: {e}")
            return []

        preferred_hostapi = _preferred_hostapi()

        has_any_output = any(dev["max_output_channels"] > 0 for dev in devices)
        if not has_any_output:
            logging.warning("No audio output devices found; mic passthrough unavailable")
            return []

        mics = []
        for i, dev in enumerate(devices):
            if dev["max_input_channels"] <= 0:
                continue
            if dev["name"] in _VIRTUAL_INPUT_DEVICE_NAMES:
                continue
            if preferred_hostapi is not None and dev["hostapi"] != preferred_hostapi:
                continue
            mics.append({"deviceId": str(i), "label": dev["name"]})

        self._device_list = mics
        self._device_index = {d["deviceId"]: d for d in mics}
        logging.info(f"Mic devices enumerated (sounddevice): {[m['label'] for m in mics]}")
        return mics

    def get_enriched_devices(self, settings: dict | None = None) -> list[dict]:
        """Return device list merged with saved preference settings."""
        if settings is None:
            settings = self.load_settings()

        enriched = []
        for dev in self._device_list:
            label = dev["label"]
            saved = settings.get(label, {})
            enriched.append(
                {
                    "deviceId": dev["deviceId"],
                    "label": label,
                    "enabled": saved.get("enabled", False),
                    "volume": saved.get("volume", 1.0),
                }
            )
        return enriched

    def get_mic_settings_state(self) -> dict:
        """Return latency and echo-cancel state from a single settings load."""
        return _extract_mic_settings_state(self.load_settings())

    def get_latency_ms(self) -> int:
        """Get the configured mic loopback latency in milliseconds."""
        settings = self.load_settings()
        return int(settings.get("_latency_ms", _DEFAULT_LATENCY_MS))

    def set_latency_ms(self, latency_ms: int) -> dict:
        """Set mic loopback latency and restart active mics to apply it.

        Returns the updated mic settings state dict.
        """
        latency_ms = max(_MIN_LATENCY_MS, min(_MAX_LATENCY_MS, latency_ms))
        settings = self.load_settings()
        settings["_latency_ms"] = latency_ms
        self.save_settings(settings)
        logging.info(f"Mic latency set to {latency_ms}ms")
        self._restart_active_mics(settings)
        return _extract_mic_settings_state(settings)

    def get_echo_cancel(self) -> bool:
        """Whether echo cancellation is enabled for mic loopback (Linux only)."""
        settings = self.load_settings()
        return bool(settings.get("_echo_cancel", False))

    def set_echo_cancel(self, enabled: bool) -> dict:
        """Enable or disable echo cancellation and restart active mics.

        Returns the updated mic settings state dict.
        """
        settings = self.load_settings()
        settings["_echo_cancel"] = enabled
        self.save_settings(settings)
        logging.info(f"Echo cancellation {'enabled' if enabled else 'disabled'}")
        self._restart_active_mics(settings)
        return _extract_mic_settings_state(settings)

    def _restart_active_mics(self, settings: dict | None = None) -> None:
        """Deactivate and re-activate all active mics to apply changed settings."""
        if settings is None:
            settings = self.load_settings()
        echo_cancel = bool(settings.get("_echo_cancel", False))
        latency_ms = int(settings.get("_latency_ms", _DEFAULT_LATENCY_MS))
        active_snapshot = [
            (mic.device_id, self._get_active_volume(mic, settings))
            for mic in self._active_mics.values()
        ]
        self._close_all()
        for device_id, volume in active_snapshot:
            self._activate_with_settings(device_id, volume, echo_cancel, latency_ms)

    def _get_active_volume(self, mic: _ActiveMic, settings: dict | None = None) -> float:
        """Read the current volume from an active mic."""
        if mic.passthrough is not None:
            return mic.passthrough.gain
        # For pactl mics, look up saved volume from settings
        dev = self._find_device(mic.device_id)
        if dev:
            if settings is None:
                settings = self.load_settings()
            saved = settings.get(dev["label"], {})
            return saved.get("volume", 1.0)
        return 1.0

    def activate(self, device_id: str, volume: float) -> bool:
        """Start audio passthrough for a microphone device."""
        if not self.available:
            return False
        if device_id in self._active_mics:
            self.update_volume(device_id, volume)
            return True

        settings = self.load_settings()
        return self._activate_with_settings(
            device_id,
            volume,
            echo_cancel=bool(settings.get("_echo_cancel", False)),
            latency_ms=int(settings.get("_latency_ms", _DEFAULT_LATENCY_MS)),
        )

    def _activate_with_settings(
        self, device_id: str, volume: float, echo_cancel: bool, latency_ms: int
    ) -> bool:
        """Activate a mic with pre-loaded settings (avoids redundant settings loads)."""
        if device_id in self._active_mics:
            self.update_volume(device_id, volume)
            return True

        if _HAS_PACTL:
            return self._activate_pactl(
                device_id, volume, echo_cancel=echo_cancel, latency_ms=latency_ms
            )
        if _SOUNDDEVICE_AVAILABLE:
            return self._activate_sounddevice(device_id, volume, latency_ms)
        return False

    def _activate_pactl(
        self, device_id: str, volume: float, *, echo_cancel: bool, latency_ms: int
    ) -> bool:
        """Activate mic passthrough via PulseAudio module-loopback.

        When echo cancellation is enabled, loads module-echo-cancel first
        to create a processed source, then loops that through to speakers.
        """
        dev = self._find_device(device_id)
        if dev is None:
            logging.error(f"Device {device_id} not found in device list")
            return False

        source_name = dev.get("paSource", dev["label"])
        vol_percent = int(min(volume, _MAX_GAIN) * 100)
        echo_module_id = None

        # Optionally load echo cancellation
        if echo_cancel:
            echo_module_id = self._load_echo_cancel_module(source_name)
            if echo_module_id:
                source_name = f"{source_name}.echo-cancel"

        try:
            result = subprocess.run(
                [
                    "pactl",
                    "load-module",
                    "module-loopback",
                    f"source={source_name}",
                    f"latency_msec={latency_ms}",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                logging.error(f"pactl load-module failed: {result.stderr.strip()}")
                self._unload_module(echo_module_id)
                return False
            module_id = result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logging.error(f"Failed to load loopback module: {e}")
            self._unload_module(echo_module_id)
            return False

        # Set source volume
        try:
            subprocess.run(
                ["pactl", "set-source-volume", source_name, f"{vol_percent}%"],
                capture_output=True,
                timeout=5,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        active = _ActiveMic(device_id=device_id, loopback_module_id=module_id)
        active.echo_module_id = echo_module_id
        self._active_mics[device_id] = active
        logging.info(f"Mic activated (pactl loopback module {module_id}): {dev['label']}")
        return True

    def _activate_sounddevice(self, device_id: str, volume: float, latency_ms: int) -> bool:
        """Activate mic passthrough via sounddevice (Windows/macOS).

        Opens a mono input stream on the mic and an output stream on the
        default output of the same host API, joined by a _Passthrough. Both
        run at the output's rate; WASAPI's auto_convert resamples the mic.
        """
        idx = int(device_id)
        try:
            dev_info = sd.query_devices(idx)
            output_idx = _find_output_for_hostapi(dev_info["hostapi"])
            if output_idx is None:
                logging.error(f"No output device found on same host API as {dev_info['name']}")
                return False
            out_info = sd.query_devices(output_idx)
            hostapi_name = sd.query_hostapis(dev_info["hostapi"])["name"]
        except sd.PortAudioError as e:
            logging.error(f"Failed to query audio devices for mic {device_id}: {e}")
            return False

        samplerate = int(out_info["default_samplerate"])
        out_channels = min(out_info["max_output_channels"], 2)
        blocksize = samplerate * _BLOCK_MS // 1000
        passthrough = _Passthrough(
            gain=min(volume, _MAX_GAIN),
            out_channels=out_channels,
            max_queued_blocks=latency_ms // _BLOCK_MS + _DRIFT_SLACK_BLOCKS,
        )
        common = {
            "samplerate": samplerate,
            "dtype": "int16",
            "blocksize": blocksize,
            "latency": latency_ms / 1000,
        }
        if "WASAPI" in hostapi_name:
            common["extra_settings"] = sd.WasapiSettings(auto_convert=True)

        streams = []
        try:
            streams.append(
                sd.RawInputStream(device=idx, channels=1, callback=passthrough.on_input, **common)
            )
            streams.append(
                sd.RawOutputStream(
                    device=output_idx,
                    channels=out_channels,
                    callback=passthrough.on_output,
                    **common,
                )
            )
            for stream in streams:
                stream.start()
        except sd.PortAudioError as e:
            logging.error(f"Failed to activate mic {dev_info['name']}: {e}")
            _close_streams(streams)
            return False

        self._active_mics[device_id] = _ActiveMic(
            device_id=device_id,
            loopback_module_id=None,
            streams=streams,
            passthrough=passthrough,
        )
        logging.info(
            f"Mic activated ({hostapi_name}): '{dev_info['name']}' -> '{out_info['name']}' "
            f"({out_channels}ch @ {samplerate}Hz, {latency_ms}ms)"
        )
        return True

    def _load_echo_cancel_module(self, source_name: str) -> str | None:
        """Load PulseAudio module-echo-cancel for a source. Returns module ID or None."""
        try:
            result = subprocess.run(
                [
                    "pactl",
                    "load-module",
                    "module-echo-cancel",
                    f"source_name={source_name}.echo-cancel",
                    f"source_master={source_name}",
                    "aec_method=webrtc",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                logging.warning(
                    f"Echo cancel module failed (continuing without): {result.stderr.strip()}"
                )
                return None
            logging.info(f"Echo cancellation loaded for source: {source_name}")
            return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logging.warning(f"Echo cancel module unavailable: {e}")
            return None

    def _unload_module(self, module_id: str | None) -> None:
        """Unload a PulseAudio module by ID, if set."""
        if not module_id:
            return
        try:
            subprocess.run(
                ["pactl", "unload-module", module_id],
                capture_output=True,
                timeout=5,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logging.warning(f"Failed to unload module {module_id}: {e}")

    def deactivate(self, device_id: str) -> None:
        """Stop audio passthrough for a microphone device."""
        active = self._active_mics.pop(device_id, None)
        if not active:
            return

        if active.loopback_module_id is not None:
            self._unload_module(active.loopback_module_id)
            self._unload_module(active.echo_module_id)
        else:
            _close_streams(active.streams)

        logging.info(f"Mic deactivated: device={device_id}")

    def update_volume(self, device_id: str, volume: float) -> None:
        """Update gain for an active microphone."""
        active = self._active_mics.get(device_id)
        if not active:
            return

        gain = min(volume, _MAX_GAIN)

        if active.loopback_module_id is not None:
            # Update volume via pactl
            dev = self._find_device(device_id)
            if dev:
                source_name = dev.get("paSource", dev["label"])
                vol_percent = int(gain * 100)
                try:
                    subprocess.run(
                        ["pactl", "set-source-volume", source_name, f"{vol_percent}%"],
                        capture_output=True,
                        timeout=5,
                    )
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    pass
        elif active.passthrough is not None:
            active.passthrough.gain = gain

        logging.debug(f"Mic volume updated: device={device_id} vol={gain}")

    def rescan(self) -> list[dict]:
        """Re-read the device list and restart passthrough to match it.

        All mics are closed first: PortAudio only sees new devices after a
        re-init, which would cut any open stream anyway, and an unplugged
        mic's stream has to go regardless.
        """
        self._close_all()
        self.enumerate_devices()
        self._activate_saved_mics()
        return self.get_enriched_devices()

    def start(self) -> None:
        """Enumerate devices, re-enable saved mics and, on Windows, watch for hot-plug."""
        if not self.available:
            return
        self.enumerate_devices()
        self._activate_saved_mics()

        if _SOUNDDEVICE_AVAILABLE and is_windows():
            self._watching = True
            threading.Thread(target=self._watch_hotplug, daemon=True).start()

    def stop(self) -> None:
        """Stop watching for devices and close all active mic streams."""
        self._watching = False
        self._close_all()

    def _close_all(self) -> None:
        for device_id in list(self._active_mics.keys()):
            self.deactivate(device_id)

    def _activate_saved_mics(self) -> None:
        """Activate every enabled mic, first switching on any new USB mics."""
        settings = self.load_settings()
        if _auto_enable_new_mics(self._device_list, settings):
            self.save_settings(settings)

        for dev in self._device_list:
            label = dev["label"]
            saved = settings.get(label, {})
            if saved.get("enabled", False):
                volume = saved.get("volume", 1.0)
                if self.activate(dev["deviceId"], volume):
                    logging.info(f"Auto-activated saved mic: {label}")

    def _watch_hotplug(self) -> None:
        """Rescan when Windows reports a device was added or removed.

        Polls MME's device counts, which Windows keeps current without the
        PortAudio re-init that a rescan costs.
        """
        last_counts = _windows_device_counts()
        while self._watching:
            time.sleep(_HOTPLUG_POLL_SECONDS)
            counts = _windows_device_counts()
            if counts == last_counts or not self._watching:
                continue
            last_counts = counts
            logging.info("Audio devices changed; rescanning microphones")
            self.rescan()
            self._events.emit("mic_devices_changed")

    def _find_device(self, device_id: str) -> dict | None:
        """Find a device dict by its ID."""
        return self._device_index.get(device_id)

    def load_settings(self) -> dict:
        """Load and parse mic settings from preferences."""
        raw = self._preferences.get_or_default("mic_settings") or "{}"
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logging.warning("mic_settings preference is malformed JSON; resetting")
            return {}

    def save_settings(self, settings: dict) -> None:
        """Persist mic settings to preferences."""
        self._preferences.set("mic_settings", json.dumps(settings))


def _extract_mic_settings_state(settings: dict) -> dict:
    """Extract latency and echo-cancel state from a settings dict."""
    return {
        "latency_ms": int(settings.get("_latency_ms", _DEFAULT_LATENCY_MS)),
        "echo_cancel": bool(settings.get("_echo_cancel", False)),
    }


def _should_auto_enable(label: str) -> bool:
    """Whether a never-seen mic should switch itself on.

    Karaoke and game mics present as USB audio. Built-in laptop mics and
    webcams (USB too, but never sung into) stay off so they cannot feed back.
    """
    lowered = label.lower()
    return "usb" in lowered and "cam" not in lowered


def _auto_enable_new_mics(devices: list[dict], settings: dict) -> bool:
    """Enable USB mics with no saved state. Returns True if settings changed.

    The choice is saved, so switching a mic off later sticks.
    """
    changed = False
    for dev in devices:
        label = dev["label"]
        if label not in settings and _should_auto_enable(label):
            settings[label] = {"enabled": True, "volume": 1.0}
            logging.info(f"New USB mic detected, enabling: {label}")
            changed = True
    return changed


def _mix_block(data: bytes, gain: float, out_channels: int) -> bytes:
    """Apply gain to a mono int16 block and spread it across the output channels."""
    samples = array("h", data)
    if gain != 1.0:
        samples = array("h", (max(-32768, min(32767, int(s * gain))) for s in samples))
    if out_channels == 1:
        return samples.tobytes()
    out = array("h", bytes(len(samples) * 2 * out_channels))
    for channel in range(out_channels):
        out[channel::out_channels] = samples
    return out.tobytes()


def _close_streams(streams: list) -> None:
    for stream in streams:
        try:
            stream.stop()
            stream.close()
        except (sd.PortAudioError, OSError) as e:
            logging.warning(f"Error closing mic stream: {e}")


def _windows_device_counts() -> tuple[int, int] | None:
    """Count of MME input and output devices, or None off Windows."""
    try:
        winmm = ctypes.windll.winmm  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return None
    return winmm.waveInGetNumDevs(), winmm.waveOutGetNumDevs()


def _preferred_hostapi() -> int | None:
    """WASAPI where present (Windows), else the default output's host API.

    PortAudio defaults to MME on Windows, whose buffering puts an audible
    echo between singing and hearing yourself; WASAPI shared mode does not.
    """
    try:
        for i, api in enumerate(sd.query_hostapis()):
            if "WASAPI" in api["name"] and api["default_output_device"] >= 0:
                return i
        return sd.query_devices(kind="output")["hostapi"]
    except (sd.PortAudioError, ValueError):
        return None


def _reinit_portaudio() -> None:
    """Force PortAudio to re-scan the system device list."""
    if not _SOUNDDEVICE_AVAILABLE:
        return
    try:
        sd._terminate()
        sd._initialize()
    except (sd.PortAudioError, OSError) as e:
        logging.warning(f"PortAudio re-init failed: {e}")


def _find_output_for_hostapi(hostapi_index: int) -> int | None:
    """Find the default output device on the given host API."""
    if not _SOUNDDEVICE_AVAILABLE:
        return None
    try:
        hostapi_info = sd.query_hostapis(hostapi_index)
        default_output = hostapi_info.get("default_output_device", -1)
        if default_output >= 0:
            return default_output
    except sd.PortAudioError:
        pass
    for i, dev in enumerate(sd.query_devices()):
        if dev["hostapi"] == hostapi_index and dev["max_output_channels"] > 0:
            return i
    return None

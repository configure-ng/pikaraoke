"""Tests for SoundManager gating, auto-enable and the sounddevice passthrough."""

from array import array

import pytest

from pikaraoke.lib import sound_manager as sm
from pikaraoke.lib.events import EventSystem
from pikaraoke.lib.preference_manager import PreferenceManager


@pytest.fixture(autouse=True)
def no_real_devices(monkeypatch):
    """Keep start() from watching, and helpers from calling, the host's real pactl."""
    monkeypatch.setattr(sm, "_device_fingerprint", lambda: None)
    monkeypatch.setattr(sm, "_pactl", lambda *args: None)


def _make_manager(tmp_path, enabled):
    preferences = PreferenceManager(config_file_path=str(tmp_path / "config.ini"))
    return sm.SoundManager(preferences=preferences, events=EventSystem(), enabled=enabled)


def test_unavailable_when_disabled_even_with_backend(tmp_path, monkeypatch):
    """available is False when the feature flag is off, regardless of backend."""
    monkeypatch.setattr(sm, "_HAS_PACTL", True)
    monkeypatch.setattr(sm, "_SOUNDDEVICE_AVAILABLE", True)

    manager = _make_manager(tmp_path, enabled=False)

    assert manager.available is False


def test_available_when_enabled_with_backend(tmp_path, monkeypatch):
    """available is True when enabled and a backend is present."""
    monkeypatch.setattr(sm, "_HAS_PACTL", True)
    monkeypatch.setattr(sm, "_SOUNDDEVICE_AVAILABLE", False)

    manager = _make_manager(tmp_path, enabled=True)

    assert manager.available is True


def test_unavailable_when_enabled_without_backend(tmp_path, monkeypatch):
    """available is False when enabled but no backend is present."""
    monkeypatch.setattr(sm, "_HAS_PACTL", False)
    monkeypatch.setattr(sm, "_SOUNDDEVICE_AVAILABLE", False)

    manager = _make_manager(tmp_path, enabled=True)

    assert manager.available is False


def test_activate_refused_when_disabled(tmp_path, monkeypatch):
    """activate() refuses to start a stream when the feature is disabled."""
    monkeypatch.setattr(sm, "_HAS_PACTL", False)
    monkeypatch.setattr(sm, "_SOUNDDEVICE_AVAILABLE", True)

    manager = _make_manager(tmp_path, enabled=False)

    assert manager.activate("0", 1.0) is False


def test_enumerate_empty_when_disabled(tmp_path, monkeypatch):
    """enumerate_devices() returns nothing when the feature is disabled."""
    monkeypatch.setattr(sm, "_HAS_PACTL", True)
    monkeypatch.setattr(sm, "_SOUNDDEVICE_AVAILABLE", True)

    manager = _make_manager(tmp_path, enabled=False)

    assert manager.enumerate_devices() == []


def test_start_noop_when_disabled(tmp_path, monkeypatch):
    """start() does not enumerate devices when the feature is disabled."""
    monkeypatch.setattr(sm, "_HAS_PACTL", True)
    monkeypatch.setattr(sm, "_SOUNDDEVICE_AVAILABLE", True)

    manager = _make_manager(tmp_path, enabled=False)
    called = False

    def fail():
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(manager, "enumerate_devices", fail)
    manager.start()

    assert called is False


def test_start_reenables_a_saved_mic(tmp_path, monkeypatch):
    """An enabled mic comes back at startup, with nobody logged in.

    The mic is a server-side loopback restored by start(), long before any
    browser connects, so gating the socket handlers cannot reach it. A host
    who switched a mic on keeps it whether or not anyone has a session.
    """
    monkeypatch.setattr(sm, "_HAS_PACTL", True)
    monkeypatch.setattr(sm, "_SOUNDDEVICE_AVAILABLE", True)
    manager = _make_manager(tmp_path, enabled=True)

    device = {"label": "USB Mic", "deviceId": "3"}
    monkeypatch.setattr(manager, "enumerate_devices", lambda: manager._device_list)
    manager._device_list = [device]
    monkeypatch.setattr(
        manager, "load_settings", lambda: {"USB Mic": {"enabled": True, "volume": 0.6}}
    )
    activated = []
    monkeypatch.setattr(manager, "activate", lambda d, v: bool(activated.append((d, v))) or True)

    manager.start()

    assert activated == [("3", 0.6)]


def _start_with_devices(tmp_path, monkeypatch, devices, saved):
    """Run start() against a fake device list; return (manager, activated)."""
    monkeypatch.setattr(sm, "_HAS_PACTL", True)
    manager = _make_manager(tmp_path, enabled=True)
    manager.save_settings(saved)
    monkeypatch.setattr(manager, "enumerate_devices", lambda: manager._device_list)
    manager._device_list = devices
    activated = []
    monkeypatch.setattr(manager, "activate", lambda d, v: bool(activated.append((d, v))) or True)
    manager.start()
    return manager, activated


def test_start_auto_enables_a_new_usb_mic(tmp_path, monkeypatch):
    """A never-seen USB mic is switched on and remembered, so it just works."""
    devices = [{"label": "Microphone (USB Microphone)", "deviceId": "4"}]

    manager, activated = _start_with_devices(tmp_path, monkeypatch, devices, {})

    assert activated == [("4", 1.0)]
    assert manager.load_settings() == {
        "Microphone (USB Microphone)": {"enabled": True, "volume": 1.0}
    }


def test_start_leaves_builtin_and_webcam_mics_off(tmp_path, monkeypatch):
    """Mics nobody sings into stay off, so they cannot feed back through the speakers."""
    devices = [
        {"label": "Microphone Array (Realtek(R) Audio)", "deviceId": "1"},
        {"label": "Microphone (USB Webcam C270)", "deviceId": "2"},
    ]

    manager, activated = _start_with_devices(tmp_path, monkeypatch, devices, {})

    assert activated == []
    assert manager.load_settings() == {}


def test_start_respects_a_usb_mic_switched_off(tmp_path, monkeypatch):
    """Auto-enable applies only to unseen mics; a host's off switch sticks."""
    devices = [{"label": "Microphone (USB Microphone)", "deviceId": "4"}]
    saved = {"Microphone (USB Microphone)": {"enabled": False, "volume": 0.5}}

    _, activated = _start_with_devices(tmp_path, monkeypatch, devices, saved)

    assert activated == []


def test_rescan_closes_mics_before_reactivating(tmp_path, monkeypatch):
    """Device ids shift on replug, so every stream is closed and reopened by label."""
    devices = [{"label": "USB Mic", "deviceId": "7"}]
    manager, _ = _start_with_devices(
        tmp_path, monkeypatch, devices, {"USB Mic": {"enabled": True, "volume": 0.8}}
    )
    manager._active_mics["3"] = sm._ActiveMic(device_id="3", loopback_module_id=None)
    order = []
    monkeypatch.setattr(manager, "deactivate", lambda d: order.append(("close", d)))
    monkeypatch.setattr(manager, "activate", lambda d, v: bool(order.append(("open", d))) or True)

    result = manager.rescan()

    assert order == [("close", "3"), ("open", "7")]
    assert result == [{"deviceId": "7", "label": "USB Mic", "enabled": True, "volume": 0.8}]


def _pcm(*samples: int) -> bytes:
    return array("h", samples).tobytes()


def test_mix_block_duplicates_mono_to_stereo():
    assert sm._mix_block(_pcm(1, -2), 1.0, 2) == _pcm(1, 1, -2, -2)


def test_mix_block_boosts_with_clipping():
    assert sm._mix_block(_pcm(100, 30000, -30000), 2.0, 1) == _pcm(200, 32767, -32768)


def test_passthrough_carries_input_to_output():
    passthrough = sm._Passthrough(gain=0.5, out_channels=2, max_queued_blocks=4)
    out = bytearray(8)

    passthrough.on_input(_pcm(10, 20), 2, None, None)
    passthrough.on_output(out, 2, None, None)

    assert bytes(out) == _pcm(5, 5, 10, 10)


def test_passthrough_plays_silence_when_mic_falls_behind():
    passthrough = sm._Passthrough(gain=1.0, out_channels=1, max_queued_blocks=4)
    out = bytearray(6)

    passthrough.on_input(_pcm(7), 1, None, None)
    passthrough.on_output(out, 3, None, None)

    assert bytes(out) == _pcm(7, 0, 0)


def test_passthrough_drops_oldest_audio_when_mic_runs_ahead():
    """Bounded backlog keeps clock drift from building into ever-growing delay."""
    passthrough = sm._Passthrough(gain=1.0, out_channels=1, max_queued_blocks=2)
    out = bytearray(4)

    for sample in (1, 2, 3):
        passthrough.on_input(_pcm(sample), 1, None, None)
    passthrough.on_output(out, 2, None, None)

    assert bytes(out) == _pcm(2, 3)


def test_hotplug_rescans_and_announces_only_on_change(tmp_path, monkeypatch):
    """A steady device count costs nothing; a plug or unplug rescans once and tells browsers."""
    manager = _make_manager(tmp_path, enabled=True)
    counts = iter([(1, 1), (1, 1), (2, 1), (2, 1)])

    def next_counts():
        try:
            return next(counts)
        except StopIteration:
            manager._watching = False
            return (2, 1)

    monkeypatch.setattr(sm, "_device_fingerprint", next_counts)
    monkeypatch.setattr(sm.time, "sleep", lambda _: None)
    rescans = []
    monkeypatch.setattr(manager, "rescan", lambda: rescans.append(1))
    announced = []
    manager._events.on("mic_devices_changed", lambda: announced.append(1))

    manager._watching = True
    manager._watch_hotplug()

    assert rescans == [1]
    assert announced == [1]


def test_latency_change_keeps_watching_for_hotplug(tmp_path, monkeypatch):
    """Reopening mics to apply a setting must not end the hot-plug watcher."""
    monkeypatch.setattr(sm, "_HAS_PACTL", True)
    manager = _make_manager(tmp_path, enabled=True)
    manager._watching = True

    manager.set_latency_ms(50)

    assert manager._watching is True


_PACTL_SOURCES = """Source #60
\tName: alsa_output.usb-Topping_DX5-00.sink.monitor
\tDescription: Monitor of DX5 II Headphones
\tProperties:
\t\tdevice.bus = "usb"
Source #62
\tName: alsa_input.usb-Blue_Yeti-00.analog-stereo
\tDescription: Yeti Stereo Microphone Analog Stereo
\tProperties:
\t\tdevice.bus = "usb"
\t\tdevice.form_factor = "microphone"
Source #63
\tName: alsa_input.usb-Logitech_C920-02.analog-stereo
\tDescription: HD Pro Webcam C920 Analog Stereo
\tProperties:
\t\tdevice.bus = "usb"
\t\tdevice.form_factor = "webcam"
Source #64
\tName: alsa_input.pci-0000_00_1f.3.analog-stereo
\tDescription: Built-in Audio Analog Stereo
\tProperties:
\t\tdevice.bus = "pci"
Source #65
\tName: alsa_input.usb-Blue_Yeti-00.analog-stereo.echo-cancel
\tDescription: Yeti (echo cancelled)
"""


def test_pactl_sources_auto_enable_by_bus_not_name(monkeypatch):
    """PipeWire's labels rarely say USB, so its bus and form factor decide.

    Monitors and our own echo-cancel sources are not mics at all.
    """
    monkeypatch.setattr(sm, "_pactl", lambda *args: _PACTL_SOURCES)

    sources = {s["label"]: s["autoEnable"] for s in sm._pactl_list_sources()}

    assert sources == {
        "Yeti Stereo Microphone Analog Stereo": True,
        "HD Pro Webcam C920 Analog Stereo": False,
        "Built-in Audio Analog Stereo": False,
    }


def test_pactl_source_names_ignore_our_own_sources(monkeypatch):
    """Loading echo-cancel adds a source; that must not read as a hot-plug and loop."""
    short = (
        "62\talsa_input.usb-Yeti\tPipeWire\ts16le 2ch 48000Hz\tRUNNING\n"
        "65\talsa_input.usb-Yeti.echo-cancel\tPipeWire\ts16le 2ch 48000Hz\tRUNNING\n"
        "60\talsa_output.usb-DX5.monitor\tPipeWire\ts32le 2ch 48000Hz\tIDLE\n"
    )
    monkeypatch.setattr(sm, "_pactl", lambda *args: short)

    assert sm._pactl_source_names() == ("alsa_input.usb-Yeti",)


def test_pactl_volume_sets_the_loopback_not_the_system_mic(tmp_path, monkeypatch):
    """The mic slider changes what PiKaraoke plays, not the mic level other apps record."""
    monkeypatch.setattr(sm, "_HAS_PACTL", True)
    sink_inputs = "Sink Input #300\n\tDriver: PipeWire\n\tOwner Module: 42\n"
    calls = []

    def fake_pactl(*args):
        calls.append(args)
        return {"load-module": "42\n", "list": sink_inputs}.get(args[0], "")

    monkeypatch.setattr(sm, "_pactl", fake_pactl)
    manager = _make_manager(tmp_path, enabled=True)
    manager._device_list = [{"deviceId": "62", "label": "Yeti", "paSource": "yeti"}]
    manager._device_index = {"62": manager._device_list[0]}

    assert manager.activate("62", 1.5)
    manager.update_volume("62", 0.5)

    volume_calls = [c for c in calls if "volume" in c[0]]
    assert volume_calls == [
        ("set-sink-input-volume", "300", "150%"),
        ("set-sink-input-volume", "300", "50%"),
    ]


class _FakeThread:
    started: list = []

    def __init__(self, target, args=(), daemon=None):
        self.target, self.args = target, args

    def start(self):
        _FakeThread.started.append(self)


@pytest.fixture
def capturing(tmp_path, monkeypatch):
    """A manager with mic passthrough available and background threads held back."""
    monkeypatch.setattr(sm, "_HAS_PACTL", True)
    _FakeThread.started = []
    monkeypatch.setattr(sm.threading, "Thread", _FakeThread)
    return _make_manager(tmp_path, enabled=True)


def _deliver_once(manager, monkeypatch):
    """Run the capture delivery loop for a single batch."""
    monkeypatch.setattr(sm.time, "sleep", lambda _: setattr(manager, "_capture_generation", -1))
    deliver = next(t for t in _FakeThread.started if t.target == manager._deliver_capture)
    deliver.target(*deliver.args)


def test_capture_copies_a_streamed_mic_to_the_score_keeper(capturing, monkeypatch):
    """Windows/macOS: the passthrough's own input is copied, no second recording."""
    passthrough = sm._Passthrough(gain=1.0, out_channels=2, max_queued_blocks=4)
    capturing._active_mics["5"] = sm._ActiveMic(
        device_id="5",
        loopback_module_id=None,
        passthrough=passthrough,
        label="USB Mic",
        samplerate=48000,
    )
    received = []

    capturing.set_capture(lambda *args: received.append(args))
    passthrough.on_input(_pcm(1, 2), 2, None, None)
    passthrough.on_input(_pcm(3), 1, None, None)
    _deliver_once(capturing, monkeypatch)

    assert received == [("USB Mic", _pcm(1, 2, 3), 48000)]


def test_capture_records_a_pactl_mic_from_its_loopback_source(capturing, monkeypatch):
    """Linux: parec records the source the loopback plays, echo-cancelled if that is on."""
    launched = []

    class FakeProcess:
        def __init__(self, cmd, **kwargs):
            launched.append(cmd)
            self.terminated = False

        def terminate(self):
            self.terminated = True

        def wait(self, timeout):
            return 0

    monkeypatch.setattr(sm.subprocess, "Popen", FakeProcess)
    capturing._active_mics["62"] = sm._ActiveMic(
        device_id="62", loopback_module_id="42", label="Yeti", capture_source="yeti.echo-cancel"
    )

    capturing.set_capture(lambda *args: None)
    process = capturing._taps["62"].process
    capturing.set_capture(None)

    assert launched == [
        [
            "parec",
            "--device=yeti.echo-cancel",
            "--raw",
            "--format=s16le",
            "--rate=16000",
            "--channels=1",
            "--latency-msec=50",
        ]
    ]
    assert process.terminated and capturing._taps == {}


def test_a_mic_plugged_in_mid_song_joins_the_capture(capturing, monkeypatch):
    capturing.set_capture(lambda *args: None)
    passthrough = sm._Passthrough(gain=1.0, out_channels=1, max_queued_blocks=4)

    def plug_in(device_id, volume, **settings):
        capturing._active_mics[device_id] = sm._ActiveMic(
            device_id=device_id, loopback_module_id=None, passthrough=passthrough, label="New"
        )
        return True

    monkeypatch.setattr(capturing, "_activate_pactl", plug_in)
    capturing.activate("9", 1.0)

    assert passthrough.tap is not None
    capturing.deactivate("9")
    assert passthrough.tap is None and capturing._taps == {}


def test_capture_is_a_no_op_without_mic_support(tmp_path, monkeypatch):
    _FakeThread.started = []
    monkeypatch.setattr(sm.threading, "Thread", _FakeThread)
    manager = _make_manager(tmp_path, enabled=False)

    manager.set_capture(lambda *args: None)

    assert _FakeThread.started == []

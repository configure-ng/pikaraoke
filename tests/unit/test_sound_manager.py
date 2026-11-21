"""Tests for SoundManager gating, auto-enable and the sounddevice passthrough."""

from array import array

from pikaraoke.lib import sound_manager as sm
from pikaraoke.lib.events import EventSystem
from pikaraoke.lib.preference_manager import PreferenceManager


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

    monkeypatch.setattr(sm, "_windows_device_counts", next_counts)
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

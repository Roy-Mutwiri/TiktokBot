"""Shared speaker output: device override plus stale-stream recovery.

Both sound paths used to open one sounddevice OutputStream on the system
default device and cache it for the life of the process. When that device went
away - a Bluetooth speaker dropping off, a virtual cable install reshuffling the
defaults - the cached stream kept accepting writes without ever raising, so the
studio went silent for the rest of the session with nothing in any log.

This module fixes the two halves of that:

* AVATAR_OUTPUT_DEVICE pins output to a chosen device instead of whatever
  Windows currently calls "default". Accepts an index ("12") or any part of a
  device name ("Odyssey", "Voicemeeter Input"); case-insensitive.
* ManagedOutputStream notices when the device it opened against is no longer
  the one it should be using, and reopens instead of writing into a dead
  handle. Failures raise AudioOutputError rather than passing silently.

What this cannot do: a speaker that is powered off but still connected accepts
audio at full level from the OS point of view. No API call distinguishes that
from a working speaker, so pin a different device or fix the speaker.
"""

import os
import threading
import time


DEVICE_ENV = "AVATAR_OUTPUT_DEVICE"
# query_devices() is cheap (~0.1 ms), so the identity check can run often.
DEVICE_RECHECK_SECONDS = 2.0
# Opening a stream is NOT cheap: ~395 ms measured here, against a ~21 ms audio
# block period. Retrying an open per failed block buries the audio thread and
# produces no sound at all, so a failed device is parked for this long and
# writes fail instantly instead.
REOPEN_BACKOFF_SECONDS = 5.0


class AudioOutputError(RuntimeError):
    pass


def _sd():
    import sounddevice as sd
    return sd


def resolve_output_device(status_callback=None):
    """Device index from AVATAR_OUTPUT_DEVICE, or None for the system default."""
    raw = os.environ.get(DEVICE_ENV, "").strip().strip('"')
    if not raw:
        return None
    sd = _sd()
    try:
        devices = sd.query_devices()
    except Exception as exc:
        _status(status_callback, f"could not list audio devices ({exc})")
        return None
    if raw.lstrip("-").isdigit():
        index = int(raw)
        if 0 <= index < len(devices) and devices[index]["max_output_channels"] > 0:
            return index
        _status(
            status_callback,
            f"{DEVICE_ENV}={raw} is not a playback device; using the default")
        return None
    needle = raw.lower()
    for index, dev in enumerate(devices):
        if dev["max_output_channels"] <= 0:
            continue
        if needle in str(dev["name"]).lower():
            return index
    _status(
        status_callback,
        f"{DEVICE_ENV}={raw} matched no playback device; using the default")
    return None


def describe_output(device=None):
    """Readable name for a device index, or for the current default."""
    try:
        sd = _sd()
        info = (sd.query_devices(device) if device is not None
                else sd.query_devices(kind="output"))
        api = sd.query_hostapis(info["hostapi"])["name"]
        return f"{info['name']} [{api}]"
    except Exception as exc:
        return f"unknown output device ({exc})"


def _device_key(device):
    """Identity of the device a stream should be on right now.

    For a pinned index this is the index itself. For the default it is the
    device name, because the index behind "default" changes as soon as anything
    is plugged in, unplugged, or installed.
    """
    if device is not None:
        return ("index", device)
    try:
        return ("default", str(_sd().query_devices(kind="output")["name"]))
    except Exception:
        return ("default", None)


def output_health(status_callback=None):
    """One-line summary of where sound is going. Useful at startup."""
    device = resolve_output_device(status_callback)
    pinned = device is not None
    return (
        f"audio out: {describe_output(device)}"
        f"{' (pinned by ' + DEVICE_ENV + ')' if pinned else ' (system default)'}")


class ManagedOutputStream:
    """An OutputStream that reopens itself when its device changes underneath."""

    def __init__(self, samplerate, channels=1, dtype="float32", blocksize=960,
                 latency=None, status_callback=None):
        self.samplerate = int(samplerate)
        self.channels = int(channels)
        self.dtype = dtype
        self.blocksize = int(blocksize)
        self.latency = latency
        self.status_callback = status_callback
        self._stream = None
        self._device = None
        self._key = None
        self._checked_at = 0.0
        self._retry_at = 0.0
        self._last_error = ""
        self._failing = False
        self._lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------
    def _open(self):
        sd = _sd()
        device = resolve_output_device(self.status_callback)
        kwargs = dict(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype=self.dtype,
            blocksize=self.blocksize,
            device=device,
        )
        if self.latency is not None:
            kwargs["latency"] = self.latency
        try:
            stream = sd.OutputStream(**kwargs)
            stream.start()
        except Exception as exc:
            raise AudioOutputError(
                f"could not open {describe_output(device)} ({exc})") from exc
        self._stream = stream
        self._device = device
        self._key = _device_key(device)
        self._checked_at = time.monotonic()
        _status(self.status_callback, f"audio output open: {describe_output(device)}")

    def _close_stream(self):
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.stop()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass

    def abort(self):
        """Drop audio already queued in the device (used by mute/interrupt).

        Callers wrap this in a bare except, so a missing method here would be
        swallowed and muting would quietly stop discarding queued speech.
        """
        with self._lock:
            stream = self._stream
            if stream is not None:
                try:
                    stream.abort()
                except Exception:
                    pass
            self._close_stream()
            self._key = None

    def open(self):
        """Open now so an unusable device fails loudly at startup, not per block."""
        with self._lock:
            self._ensure_open()

    def start(self):
        """No-op: unmuting is enough, the next write reopens the stream.

        abort() closes the underlying stream, so there is nothing to restart -
        and reopening here would pick a device before we know it is needed.
        """
        return

    def stop(self):
        with self._lock:
            stream = self._stream
            if stream is not None:
                try:
                    stream.stop()
                except Exception:
                    pass

    def close(self):
        with self._lock:
            self._close_stream()
            self._key = None

    # -- writing -----------------------------------------------------------
    def _device_changed(self):
        now = time.monotonic()
        if now - self._checked_at < DEVICE_RECHECK_SECONDS:
            return False
        self._checked_at = now
        key = _device_key(self._device)
        if key == self._key:
            return False
        # A None name means the query failed outright; treat that as a blip
        # rather than tearing down a stream that may still be fine.
        if key[0] == "default" and key[1] is None:
            return False
        _status(
            self.status_callback,
            f"playback device changed ({self._key[1]} -> {key[1]}); reopening")
        return True

    # -- failure bookkeeping ----------------------------------------------
    def _park(self, error):
        """Stop using this device for a while and say so once."""
        self._close_stream()
        self._retry_at = time.monotonic() + REOPEN_BACKOFF_SECONDS
        self._last_error = str(error)
        if not self._failing:
            self._failing = True
            _status(
                self.status_callback,
                f"AUDIO OUTPUT FAILING: {error} - retrying every "
                f"{int(REOPEN_BACKOFF_SECONDS)}s")

    def _note_working(self):
        if self._failing:
            self._failing = False
            _status(self.status_callback, "audio output recovered")

    @property
    def healthy(self):
        return not self._failing

    @property
    def last_error(self):
        return self._last_error

    def _ensure_open(self):
        """Open the stream, honouring the backoff. Raises if unavailable."""
        if self._stream is not None:
            return
        if time.monotonic() < self._retry_at:
            # Cheap failure: no 395 ms open attempt inside the backoff window.
            raise AudioOutputError(
                self._last_error or "audio output unavailable")
        try:
            self._open()
        except AudioOutputError as exc:
            self._park(exc)
            raise

    def write(self, block):
        """Write one block. Reopens on device change; backs off on failure.

        Raises AudioOutputError when the block could not be played, so callers
        that have a fallback (the TTS winsound path) can use it. The raise is
        always cheap - a dead device is parked, never reopened per block.
        """
        with self._lock:
            if self._stream is not None and self._device_changed():
                self._close_stream()
            self._ensure_open()
            try:
                self._stream.write(block)
                self._note_working()
                return
            except Exception as exc:
                # One immediate reopen handles the device being swapped
                # mid-write; anything worse gets parked rather than retried.
                self._close_stream()
                if time.monotonic() < self._retry_at:
                    self._park(exc)
                    raise AudioOutputError(str(exc)) from exc
                _status(
                    self.status_callback,
                    f"playback write failed ({exc}); reopening output once")
            try:
                self._open()
                self._stream.write(block)
                self._note_working()
            except Exception as exc:
                self._park(exc)
                raise AudioOutputError(
                    f"playback failed after reopening ({exc})") from exc


def _status(callback, message):
    if callback is None:
        return
    try:
        callback(message)
    except Exception:
        pass


def _main(argv):
    """`python engines/audio_output.py [list|test <device>]`"""
    import numpy as np

    sd = _sd()
    action = argv[1] if len(argv) > 1 else "list"
    if action == "list":
        print(output_health(print))
        print()
        for index, dev in enumerate(sd.query_devices()):
            if dev["max_output_channels"] <= 0:
                continue
            api = sd.query_hostapis(dev["hostapi"])["name"]
            print(f"{index:3d}  {dev['name'][:44]:<44} {api}")
        print(f"\nPin one with: set {DEVICE_ENV}=<index or name fragment>")
        return 0

    if action == "test":
        target = argv[2] if len(argv) > 2 else ""
        if target:
            os.environ[DEVICE_ENV] = target
        device = resolve_output_device(print)
        print(f"playing a 2s tone on {describe_output(device)}")
        rate = 48000
        t = np.arange(int(rate * 2.0)) / rate
        tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype("float32")
        out = ManagedOutputStream(rate, status_callback=print)
        try:
            for i in range(0, len(tone), 960):
                out.write(tone[i:i + 960].reshape(-1, 1))
        finally:
            out.close()
        print("done - if you heard nothing, that device is not your speaker")
        return 0

    print(__doc__)
    return 1


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv))

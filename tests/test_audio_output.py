import os
import sys
import types
import unittest
from unittest import mock


ENGINES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "engines")
if ENGINES_DIR not in sys.path:
    sys.path.insert(0, ENGINES_DIR)

import audio_output
from audio_output import (
    AudioOutputError,
    ManagedOutputStream,
    describe_output,
    resolve_output_device,
)


DEVICES = [
    {"name": "Microphone (webcam)", "max_output_channels": 0, "hostapi": 0},
    {"name": "Speakers (2- Onyx Studio 9)", "max_output_channels": 2, "hostapi": 0},
    {"name": "Voicemeeter Input (VB-Audio)", "max_output_channels": 8, "hostapi": 0},
    {"name": "Odyssey G93SD (NVIDIA)", "max_output_channels": 2, "hostapi": 0},
]


class FakeStream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.closed = False
        self.aborted = False
        self.written = []
        self.fail_on_write = False

    def start(self):
        self.started = True

    def write(self, block):
        if self.fail_on_write:
            raise RuntimeError("device gone")
        self.written.append(block)

    def abort(self):
        self.aborted = True

    def stop(self):
        pass

    def close(self):
        self.closed = True


def fake_sd(default_name="Speakers (2- Onyx Studio 9)", opened=None):
    """A stand-in sounddevice module whose default output can be changed."""
    state = {"default": default_name}

    def query_devices(device=None, kind=None):
        if kind == "output":
            return {"name": state["default"], "max_output_channels": 2,
                    "hostapi": 0}
        if device is None:
            return DEVICES
        return DEVICES[device]

    def OutputStream(**kwargs):
        stream = FakeStream(**kwargs)
        if opened is not None:
            opened.append(stream)
        return stream

    module = types.SimpleNamespace(
        query_devices=query_devices,
        query_hostapis=lambda i: {"name": "MME"},
        OutputStream=OutputStream,
    )
    return module, state


class ResolveDeviceTests(unittest.TestCase):
    def setUp(self):
        os.environ.pop(audio_output.DEVICE_ENV, None)

    def test_no_override_uses_system_default(self):
        module, _ = fake_sd()
        with mock.patch.object(audio_output, "_sd", return_value=module):
            self.assertIsNone(resolve_output_device())

    def test_override_by_index(self):
        module, _ = fake_sd()
        with mock.patch.object(audio_output, "_sd", return_value=module), \
                mock.patch.dict(os.environ, {audio_output.DEVICE_ENV: "3"}):
            self.assertEqual(resolve_output_device(), 3)

    def test_override_by_name_fragment_is_case_insensitive(self):
        module, _ = fake_sd()
        with mock.patch.object(audio_output, "_sd", return_value=module), \
                mock.patch.dict(os.environ, {audio_output.DEVICE_ENV: "odyssey"}):
            self.assertEqual(resolve_output_device(), 3)

    def test_input_only_index_is_rejected(self):
        module, _ = fake_sd()
        notes = []
        with mock.patch.object(audio_output, "_sd", return_value=module), \
                mock.patch.dict(os.environ, {audio_output.DEVICE_ENV: "0"}):
            self.assertIsNone(resolve_output_device(notes.append))
        self.assertTrue(any("not a playback device" in n for n in notes))

    def test_unmatched_name_falls_back_to_default_with_a_warning(self):
        module, _ = fake_sd()
        notes = []
        with mock.patch.object(audio_output, "_sd", return_value=module), \
                mock.patch.dict(os.environ, {audio_output.DEVICE_ENV: "nosuchdev"}):
            self.assertIsNone(resolve_output_device(notes.append))
        self.assertTrue(any("matched no playback device" in n for n in notes))

    def test_describe_output_names_the_device_and_api(self):
        module, _ = fake_sd()
        with mock.patch.object(audio_output, "_sd", return_value=module):
            self.assertIn("Odyssey", describe_output(3))
            self.assertIn("MME", describe_output(3))


class ManagedStreamTests(unittest.TestCase):
    def setUp(self):
        os.environ.pop(audio_output.DEVICE_ENV, None)

    def test_first_write_opens_and_starts_the_stream(self):
        opened = []
        module, _ = fake_sd(opened=opened)
        with mock.patch.object(audio_output, "_sd", return_value=module):
            out = ManagedOutputStream(48000)
            out.write([0.0] * 960)
        self.assertEqual(len(opened), 1)
        self.assertTrue(opened[0].started)
        self.assertEqual(len(opened[0].written), 1)

    def test_default_device_change_reopens_instead_of_writing_into_a_dead_handle(self):
        opened = []
        module, state = fake_sd(opened=opened)
        notes = []
        with mock.patch.object(audio_output, "_sd", return_value=module):
            out = ManagedOutputStream(48000, status_callback=notes.append)
            out.write([0.0] * 960)
            # The Bluetooth speaker drops off and Windows moves "default".
            state["default"] = "Voicemeeter Input (VB-Audio)"
            out._checked_at = 0.0  # let the throttled recheck run
            out.write([0.0] * 960)
        self.assertEqual(len(opened), 2)
        self.assertTrue(opened[0].closed)
        self.assertEqual(len(opened[1].written), 1)
        self.assertTrue(any("device changed" in n for n in notes))

    def test_device_identity_check_is_throttled(self):
        opened = []
        module, state = fake_sd(opened=opened)
        with mock.patch.object(audio_output, "_sd", return_value=module):
            out = ManagedOutputStream(48000)
            out.write([0.0] * 960)
            state["default"] = "Voicemeeter Input (VB-Audio)"
            # No _checked_at reset: within the throttle window nothing reopens.
            for _ in range(20):
                out.write([0.0] * 960)
        self.assertEqual(len(opened), 1)

    def test_pinned_index_is_immune_to_default_device_churn(self):
        opened = []
        module, state = fake_sd(opened=opened)
        with mock.patch.object(audio_output, "_sd", return_value=module), \
                mock.patch.dict(os.environ, {audio_output.DEVICE_ENV: "3"}):
            out = ManagedOutputStream(48000)
            out.write([0.0] * 960)
            state["default"] = "Voicemeeter Input (VB-Audio)"
            out._checked_at = 0.0
            out.write([0.0] * 960)
        self.assertEqual(len(opened), 1)
        self.assertEqual(opened[0].kwargs["device"], 3)

    def test_write_failure_reopens_once_and_retries(self):
        opened = []
        module, _ = fake_sd(opened=opened)
        notes = []
        with mock.patch.object(audio_output, "_sd", return_value=module):
            out = ManagedOutputStream(48000, status_callback=notes.append)
            out.write([0.0] * 960)
            opened[0].fail_on_write = True
            out.write([0.0] * 960)
        self.assertEqual(len(opened), 2)
        self.assertEqual(len(opened[1].written), 1)
        self.assertTrue(any("write failed" in n for n in notes))

    def test_persistent_failure_raises_instead_of_going_silent(self):
        """The whole point: a dead output must be loud, not quietly mute."""
        module, _ = fake_sd()

        def always_fails(**kwargs):
            stream = FakeStream(**kwargs)
            stream.fail_on_write = True
            return stream

        module.OutputStream = always_fails
        with mock.patch.object(audio_output, "_sd", return_value=module):
            out = ManagedOutputStream(48000)
            with self.assertRaises(AudioOutputError):
                out.write([0.0] * 960)

    def test_dead_device_is_not_reopened_once_per_block(self):
        """Opening costs ~395ms vs a ~21ms block: per-block retries kill audio."""
        opens = []
        module, _ = fake_sd()

        def counting_open(**kwargs):
            opens.append(kwargs)
            raise RuntimeError("device gone")

        module.OutputStream = counting_open
        with mock.patch.object(audio_output, "_sd", return_value=module):
            out = ManagedOutputStream(48000)
            for _ in range(200):
                with self.assertRaises(AudioOutputError):
                    out.write([0.0] * 960)
        # One real attempt, then parked - not 200 attempts.
        self.assertEqual(len(opens), 1)

    def test_backoff_expiry_allows_one_fresh_attempt(self):
        opens = []
        module, _ = fake_sd()

        def counting_open(**kwargs):
            opens.append(kwargs)
            raise RuntimeError("device gone")

        module.OutputStream = counting_open
        with mock.patch.object(audio_output, "_sd", return_value=module):
            out = ManagedOutputStream(48000)
            with self.assertRaises(AudioOutputError):
                out.write([0.0] * 960)
            out._retry_at = 0.0  # pretend the backoff window elapsed
            with self.assertRaises(AudioOutputError):
                out.write([0.0] * 960)
        self.assertEqual(len(opens), 2)

    def test_failure_is_reported_once_then_recovery_is_reported(self):
        notes = []
        module, _ = fake_sd()
        broken = {"yes": True}

        def maybe_open(**kwargs):
            if broken["yes"]:
                raise RuntimeError("device gone")
            return FakeStream(**kwargs)

        module.OutputStream = maybe_open
        with mock.patch.object(audio_output, "_sd", return_value=module):
            out = ManagedOutputStream(48000, status_callback=notes.append)
            for _ in range(20):
                with self.assertRaises(AudioOutputError):
                    out.write([0.0] * 960)
            failing = [n for n in notes if "AUDIO OUTPUT FAILING" in n]
            self.assertEqual(len(failing), 1, "should warn once, not per block")
            broken["yes"] = False
            out._retry_at = 0.0
            out.write([0.0] * 960)
        self.assertTrue(any("recovered" in n for n in notes))
        self.assertTrue(out.healthy)

    def test_open_surfaces_a_dead_device_immediately(self):
        module, _ = fake_sd()

        def cannot_open(**kwargs):
            raise RuntimeError("no such device")

        module.OutputStream = cannot_open
        with mock.patch.object(audio_output, "_sd", return_value=module):
            out = ManagedOutputStream(48000)
            with self.assertRaises(AudioOutputError):
                out.open()

    def test_open_failure_raises_named_error(self):
        module, _ = fake_sd()

        def cannot_open(**kwargs):
            raise RuntimeError("Invalid sample rate")

        module.OutputStream = cannot_open
        with mock.patch.object(audio_output, "_sd", return_value=module):
            out = ManagedOutputStream(48000)
            with self.assertRaises(AudioOutputError) as ctx:
                out.write([0.0] * 960)
        self.assertIn("Invalid sample rate", str(ctx.exception))

    def test_abort_discards_queued_audio_and_forces_a_reopen(self):
        opened = []
        module, _ = fake_sd(opened=opened)
        with mock.patch.object(audio_output, "_sd", return_value=module):
            out = ManagedOutputStream(48000)
            out.write([0.0] * 960)
            out.abort()
            out.write([0.0] * 960)
        self.assertTrue(opened[0].aborted)
        self.assertTrue(opened[0].closed)
        self.assertEqual(len(opened), 2)

    def test_start_after_abort_is_a_noop_that_still_recovers_on_write(self):
        opened = []
        module, _ = fake_sd(opened=opened)
        with mock.patch.object(audio_output, "_sd", return_value=module):
            out = ManagedOutputStream(48000)
            out.write([0.0] * 960)
            out.abort()
            out.start()  # unmute path
            out.write([0.0] * 960)
        self.assertEqual(len(opened), 2)
        self.assertEqual(len(opened[1].written), 1)


if __name__ == "__main__":
    unittest.main()

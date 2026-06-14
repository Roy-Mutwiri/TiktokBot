import os
import sys
import unittest
from unittest import mock

import numpy as np


ENGINES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "engines")
if ENGINES_DIR not in sys.path:
    sys.path.insert(0, ENGINES_DIR)

from youtube_audio import (
    DEMUCS_MODEL_FILENAME,
    YouTubeAudioPlayer,
    _build_ffmpeg_cmd,
    _has_hash_prefix,
    _isolate_vocals,
    _prepare_demucs_checkpoint,
    pcm16_bytes_to_float,
)


PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


class YouTubeAudioTests(unittest.TestCase):
    def test_voice_isolation_streams_demucs_progress(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "audio.webm")
            work = os.path.join(tmp, "demucs_stems", "htdemucs", "audio")
            vocals = os.path.join(work, "vocals.wav")
            os.makedirs(work)
            with open(source, "wb") as f:
                f.write(b"audio")
            with open(vocals, "wb") as f:
                f.write(b"v" * 5000)

            class FakeStdout:
                def __init__(self, text):
                    self.text = iter(text)

                def read(self, _size):
                    return next(self.text, "")

            fake_proc = mock.Mock(
                stdout=FakeStdout(
                    "Separating 12%\rSeparating 58%\rSeparating 100%\n"))
            fake_proc.wait.return_value = 0
            statuses = []
            with mock.patch("youtube_audio._prepare_demucs_checkpoint"), \
                    mock.patch(
                        "youtube_audio.subprocess.Popen", return_value=fake_proc):
                result = _isolate_vocals(source, statuses.append)

            self.assertEqual(result, os.path.join(tmp, "vocals.wav"))
            self.assertIn(
                "voice isolation 58% - removing background music", statuses)
            self.assertIn(
                "voice isolation 100% - removing background music", statuses)

    def test_vocal_isolation_reuses_cached_stem(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "audio.webm")
            vocals = os.path.join(tmp, "vocals.wav")
            with open(source, "wb") as f:
                f.write(b"source")
            with open(vocals, "wb") as f:
                f.write(b"v" * 5000)
            self.assertEqual(_isolate_vocals(source), vocals)

    def test_pcm16_bytes_to_float(self):
        raw = np.array([-32768, 0, 32767], dtype="<i2").tobytes()
        out = pcm16_bytes_to_float(raw)

        self.assertEqual(out.dtype, np.float32)
        self.assertAlmostEqual(float(out[0]), -1.0)
        self.assertAlmostEqual(float(out[1]), 0.0)
        self.assertLess(float(out[2]), 1.0)
        self.assertGreater(float(out[2]), 0.99)

    def test_hash_prefix_accepts_verified_file(self):
        import hashlib
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.th")
            with open(path, "wb") as f:
                f.write(b"checkpoint")
            digest = hashlib.sha256(b"checkpoint").hexdigest()

            self.assertTrue(_has_hash_prefix(path, digest[:8]))
            self.assertFalse(_has_hash_prefix(path, "00000000"))

    def test_demucs_checkpoint_reuses_verified_cache(self):
        import hashlib
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            checkpoints = os.path.join(tmp, "checkpoints")
            os.makedirs(checkpoints)
            path = os.path.join(checkpoints, DEMUCS_MODEL_FILENAME)
            with open(path, "wb") as f:
                f.write(b"cached-model")
            digest = hashlib.sha256(b"cached-model").hexdigest()

            with mock.patch("youtube_audio.DEMUCS_MODEL_HASH_PREFIX", digest[:8]), \
                    mock.patch("torch.hub.get_dir", return_value=tmp), \
                    mock.patch("youtube_audio.subprocess.run") as run:
                self.assertEqual(_prepare_demucs_checkpoint(), path)
                run.assert_not_called()

    def test_demucs_checkpoint_resumes_until_hash_matches(self):
        import hashlib
        import tempfile

        payload = b"complete-model"
        digest = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            partial = os.path.join(
                tmp, "checkpoints", DEMUCS_MODEL_FILENAME + ".part")

            def fake_run(*_args, **_kwargs):
                with open(partial, "ab") as f:
                    f.write(payload if os.path.getsize(partial) else payload[:4])
                if os.path.getsize(partial) > len(payload):
                    with open(partial, "wb") as f:
                        f.write(payload)
                return mock.Mock(returncode=0, stderr="", stdout="")

            with mock.patch("youtube_audio.DEMUCS_MODEL_HASH_PREFIX", digest[:8]), \
                    mock.patch("torch.hub.get_dir", return_value=tmp), \
                    mock.patch("youtube_audio.shutil.which", return_value="curl"), \
                    mock.patch("youtube_audio.subprocess.run", side_effect=fake_run) as run:
                path = _prepare_demucs_checkpoint(attempts=2)

            self.assertEqual(run.call_count, 2)
            with open(path, "rb") as f:
                self.assertEqual(f.read(), payload)

    def test_ffmpeg_seek_is_output_side_for_accuracy(self):
        cmd = _build_ffmpeg_cmd(
            "https://example.test/audio.m4a",
            start_seconds=570,
            end_seconds=630,
        )

        input_index = cmd.index("-i")
        seek_indexes = [i for i, part in enumerate(cmd) if part == "-ss"]
        fast_seek_index, fine_seek_index = seek_indexes
        duration_index = cmd.index("-t")

        self.assertLess(fast_seek_index, input_index)
        self.assertGreater(fine_seek_index, input_index)
        self.assertGreater(duration_index, fine_seek_index)
        self.assertEqual(cmd[fast_seek_index + 1], "450.0")
        self.assertEqual(cmd[fine_seek_index + 1], "120.0")
        self.assertEqual(cmd[duration_index + 1], "60.0")

    def test_real_voice_disguise_is_applied_inside_ffmpeg(self):
        cmd = _build_ffmpeg_cmd(
            "cached-audio.webm",
            start_seconds=0,
            end_seconds=30,
            voice_disguise=True,
        )

        filter_value = cmd[cmd.index("-af") + 1]
        self.assertIn("dialoguenhance=", filter_value)
        self.assertIn("pan=mono|c0=FC", filter_value)
        self.assertIn("afftdn=", filter_value)
        self.assertIn("rubberband=tempo=1.0", filter_value)
        self.assertIn("formant=preserved", filter_value)
        self.assertIn("asetrate=48000*", filter_value)
        self.assertIn("aresample=48000", filter_value)
        self.assertIn("atempo=", filter_value)
        self.assertIn("acompressor=", filter_value)
        self.assertEqual(cmd[cmd.index("-ar") + 1], "48000")

    def test_female_persona_uses_distinct_pitch_profile(self):
        male = _build_ffmpeg_cmd(
            "cached-audio.webm", voice_disguise=True, persona="deep_male")
        woman = _build_ffmpeg_cmd(
            "cached-audio.webm", voice_disguise=True, persona="natural_woman")

        male_filter = male[male.index("-af") + 1]
        woman_filter = woman[woman.index("-af") + 1]
        self.assertIn("asetrate=48000*0.740000", male_filter)
        self.assertIn("asetrate=48000*1.250000", woman_filter)
        self.assertNotEqual(male_filter, woman_filter)

    def test_smooth_transition_ramps_in_first_audio_blocks(self):
        mouth = mock.Mock()
        player = YouTubeAudioPlayer(
            mouth, monitor=False, smooth_transition=True)
        player.fade_blocks = 2
        player.noise_floor = 0.0
        block = np.ones(12, dtype=np.float32)

        first = player._smooth_block(block)
        second = player._smooth_block(block)
        third = player._smooth_block(block)

        self.assertAlmostEqual(float(first[0]), 0.0)
        self.assertLess(float(first[-1]), 0.5)
        self.assertGreater(float(second[0]), 0.45)
        self.assertAlmostEqual(float(third[0]), 1.0)

    def test_persona_change_restarts_from_current_position(self):
        from avatar_studio import AvatarStudio

        class FakeRoot:
            def __init__(self):
                self.scheduled = []

            def after(self, delay, callback):
                self.scheduled.append((delay, callback))
                callback()

        class FakeVar:
            def get(self):
                return "Layla - natural woman"

        class FakePlayer:
            position_seconds = 193.5

            def __init__(self):
                self.stopped = False

            def stop(self):
                self.stopped = True

        studio = AvatarStudio.__new__(AvatarStudio)
        player = FakePlayer()
        calls = []
        logs = []
        studio.root = FakeRoot()
        studio.youtube_persona_var = FakeVar()
        studio._youtube_audio = player
        studio._youtube_audio_mode = True
        studio._youtube_mode = "youtube"
        studio._youtube_busy = True
        studio._log_msg = logs.append
        studio.speak_youtube_audio = (
            lambda start_override=None: calls.append(start_override))

        studio._on_youtube_persona_change()

        self.assertTrue(player.stopped)
        self.assertIsNone(studio._youtube_audio)
        self.assertFalse(studio._youtube_audio_mode)
        self.assertFalse(studio._youtube_busy)
        self.assertEqual(calls, [193.5])
        self.assertIn("03:13", logs[-1])


if __name__ == "__main__":
    unittest.main()

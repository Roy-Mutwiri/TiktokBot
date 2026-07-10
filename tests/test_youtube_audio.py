import os
import sys
import threading
import types
import unittest
from unittest import mock

import numpy as np


ENGINES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "engines")
if ENGINES_DIR not in sys.path:
    sys.path.insert(0, ENGINES_DIR)

from youtube_audio import (
    DEMUCS_MODEL_FILENAME,
    MONITOR_BLOCK,
    MONITOR_RATE,
    YouTubeAudioPlayer,
    _build_ffmpeg_cmd,
    _has_hash_prefix,
    _is_live_info,
    _isolate_vocals,
    _prepare_demucs_checkpoint,
    _resolve_audio_source,
    _select_live_audio_url,
    pcm16_bytes_to_float,
)
from youtube_dlp_options import (
    YouTubeDlpAuthError,
    extract_info_with_retries,
)


PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


class YouTubeAudioTests(unittest.TestCase):
    def test_ytdlp_retries_bot_check_with_browser_cookies(self):
        calls = []

        class FakeYDL:
            def __init__(self, opts):
                self.opts = opts
                calls.append(opts)

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def extract_info(self, _url, download):
                if "cookiesfrombrowser" not in self.opts:
                    raise RuntimeError(
                        "Sign in to confirm you're not a bot")
                return {"title": "OK", "download": download}

        fake_module = types.SimpleNamespace(YoutubeDL=FakeYDL)
        statuses = []
        with mock.patch.dict(
                os.environ,
                {"AVATAR_YOUTUBE_COOKIE_BROWSERS": "edge"},
                clear=False):
            info = extract_info_with_retries(
                fake_module, "https://youtube.test/watch?v=abc",
                {"quiet": True}, download=False,
                status_callback=statuses.append)

        self.assertEqual(info["title"], "OK")
        self.assertNotIn("cookiesfrombrowser", calls[0])
        self.assertEqual(calls[1]["cookiesfrombrowser"], ("edge",))
        self.assertIn("youtube: retrying with edge cookies", statuses)

    def test_ytdlp_auth_error_includes_cookie_hint(self):
        class FakeYDL:
            def __init__(self, _opts):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def extract_info(self, _url, download):
                raise RuntimeError("Sign in to confirm you're not a bot")

        fake_module = types.SimpleNamespace(YoutubeDL=FakeYDL)
        with mock.patch.dict(
                os.environ,
                {"AVATAR_YOUTUBE_COOKIE_BROWSERS": ""},
                clear=False):
            with self.assertRaises(YouTubeDlpAuthError) as raised:
                extract_info_with_retries(
                    fake_module, "https://youtube.test/watch?v=abc",
                    {"quiet": True})

        self.assertIn("AVATAR_YOUTUBE_COOKIES", str(raised.exception))

    def test_live_info_detection_excludes_upcoming_streams(self):
        self.assertTrue(_is_live_info({"is_live": True}))
        self.assertTrue(_is_live_info({"live_status": "is_live"}))
        self.assertFalse(_is_live_info({"live_status": "is_upcoming"}))
        self.assertFalse(_is_live_info({"live_status": "was_live"}))

    def test_live_audio_selection_prefers_audio_only_format(self):
        info = {
            "url": "https://example.test/fallback",
            "acodec": "aac",
            "vcodec": "h264",
            "formats": [
                {
                    "url": "https://example.test/video",
                    "acodec": "aac",
                    "vcodec": "h264",
                    "abr": 128,
                },
                {
                    "url": "https://example.test/audio",
                    "acodec": "opus",
                    "vcodec": "none",
                    "abr": 96,
                },
            ],
        }

        self.assertEqual(
            _select_live_audio_url(info),
            "https://example.test/audio",
        )

    def test_muted_audio_path_is_paced_in_realtime(self):
        player = YouTubeAudioPlayer.__new__(YouTubeAudioPlayer)
        player.position_blocks = 10
        player._playback_anchor_t = 100.0

        block_seconds = MONITOR_BLOCK / float(MONITOR_RATE)
        with mock.patch(
                "youtube_audio.time.monotonic",
                return_value=100.0 + 9 * block_seconds), \
                mock.patch("youtube_audio.time.sleep") as sleep:
            player._pace_realtime_if_needed(wrote_monitor=False)

        sleep.assert_called_once()
        self.assertAlmostEqual(sleep.call_args.args[0], block_seconds, places=4)

    def test_blocking_audio_write_does_not_add_extra_sleep(self):
        player = YouTubeAudioPlayer.__new__(YouTubeAudioPlayer)
        player.position_blocks = 10
        player._playback_anchor_t = 100.0

        with mock.patch("youtube_audio.time.sleep") as sleep:
            player._pace_realtime_if_needed(wrote_monitor=True)

        sleep.assert_not_called()

    def test_live_audio_resolver_does_not_download_infinite_stream(self):
        info = {
            "title": "Live show",
            "is_live": True,
            "live_status": "is_live",
            "duration": None,
            "url": "https://example.test/live.m3u8",
            "acodec": "aac",
            "vcodec": "none",
        }

        class FakeYDL:
            def __init__(self, _opts):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def extract_info(self, _url, download):
                self.download = download
                if download:
                    raise AssertionError("live streams must not be downloaded")
                return info

        fake_module = types.SimpleNamespace(YoutubeDL=FakeYDL)
        statuses = []
        with mock.patch("youtube_audio.get_cached_audio", return_value=None), \
                mock.patch.dict(sys.modules, {"yt_dlp": fake_module}):
            source, title, duration, cache_hit = _resolve_audio_source(
                "https://youtube.test/live", statuses.append)

        self.assertEqual(source, "https://example.test/live.m3u8")
        self.assertEqual(title, "Live show")
        self.assertEqual(duration, 0.0)
        self.assertFalse(cache_hit)
        self.assertIn(
            "live stream found - connecting without download", statuses)

    def test_audio_download_uses_unique_file_and_ignores_stale_part(self):
        import tempfile

        probe_info = {
            "title": "Cached show",
            "duration": 123.0,
            "is_live": False,
        }
        ydl_opts = []

        with tempfile.TemporaryDirectory() as tmp:
            stale = os.path.join(tmp, "audio.webm.part")
            with open(stale, "wb") as f:
                f.write(b"stale")

            class FakeYDL:
                def __init__(self, opts):
                    self.opts = opts
                    ydl_opts.append(opts)

                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return False

                def extract_info(self, _url, download):
                    if not download:
                        return probe_info
                    outtmpl = self.opts["outtmpl"]
                    path = outtmpl.replace("%(ext)s", "webm")
                    with open(path, "wb") as f:
                        f.write(b"audio")
                    return {
                        "title": "Cached show",
                        "duration": 123.0,
                        "requested_downloads": [{"filepath": path}],
                    }

            fake_module = types.SimpleNamespace(YoutubeDL=FakeYDL)
            saved = {}
            with mock.patch("youtube_audio.get_cached_audio", return_value=None), \
                    mock.patch("youtube_audio.audio_dir", return_value=tmp), \
                    mock.patch("youtube_audio.save_audio",
                               side_effect=lambda *args: saved.setdefault("args", args)), \
                    mock.patch.dict(sys.modules, {"yt_dlp": fake_module}):
                source, title, duration, cache_hit = _resolve_audio_source(
                    "https://youtube.test/watch?v=abc123")

            self.assertEqual(title, "Cached show")
            self.assertEqual(duration, 123.0)
            self.assertFalse(cache_hit)
            self.assertTrue(os.path.exists(source))
            self.assertFalse(os.path.exists(stale))
            self.assertFalse(os.path.basename(source).startswith("audio."))
            self.assertFalse(ydl_opts[-1]["continuedl"])
            self.assertEqual(saved["args"][3], source)

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

    def test_ffmpeg_reconnects_remote_live_streams(self):
        cmd = _build_ffmpeg_cmd("https://example.test/live.m3u8")

        self.assertIn("-reconnect", cmd)
        self.assertIn("-reconnect_streamed", cmd)
        self.assertIn("-reconnect_delay_max", cmd)
        self.assertLess(cmd.index("-reconnect"), cmd.index("-i"))

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

    def test_ready_speech_mutes_youtube_without_pausing_playback(self):
        from avatar_studio import AvatarStudio

        class FakePaused:
            def is_set(self):
                return False

        class FakePlayer:
            def __init__(self):
                self._running = True
                self._paused = FakePaused()
                self.duck_gain = 0.22
                self._target_output_gain = 1.0
                self.duck_calls = []
                self.pause_calls = 0
                self.resume_calls = 0

            def set_ducked(self, ducked, gain=None):
                self.duck_calls.append((ducked, gain))
                if gain is not None:
                    self.duck_gain = gain
                self._target_output_gain = self.duck_gain if ducked else 1.0

            def pause(self):
                self.pause_calls += 1

            def resume(self):
                self.resume_calls += 1

        class FakeTTS:
            muted = False
            _playback_match_persona = "deep_male"

            def set_playback_voice_match(self, persona):
                self._playback_match_persona = persona

            def set_muted(self, muted):
                self.muted = muted

        studio = AvatarStudio.__new__(AvatarStudio)
        player = FakePlayer()
        studio._youtube_audio = player
        studio._youtube_mode = "youtube"
        studio.tts = FakeTTS()
        studio._audio_handoff_lock = threading.Lock()
        studio._audio_handoff_token = 0
        studio._audio_handoff_state = None
        studio._log_msg = mock.Mock()

        token = studio._pause_youtube_for_ready_speech()

        self.assertEqual(player.duck_calls, [(True, 0.0)])
        self.assertEqual(player.pause_calls, 0)
        self.assertEqual(player.resume_calls, 0)
        self.assertEqual(player._target_output_gain, 0.0)

        studio._restore_youtube_after_ready_speech(token)

        self.assertEqual(player.pause_calls, 0)
        self.assertEqual(player.resume_calls, 0)
        self.assertEqual(player.duck_gain, 0.22)
        self.assertEqual(player._target_output_gain, 1.0)

    def test_failed_real_audio_is_visible_in_youtube_status(self):
        from avatar_studio import AvatarStudio

        class FakeLabel:
            def __init__(self):
                self.kwargs = {}

            def configure(self, **kwargs):
                self.kwargs.update(kwargs)

        studio = AvatarStudio.__new__(AvatarStudio)
        studio._youtube_mode = "market"
        studio._youtube_audio = None
        studio._youtube_audio_mode = False
        studio._youtube_audio_status = "failed"
        studio._youtube_chunks = []
        studio._youtube_index = 0
        studio.youtube_status_lbl = FakeLabel()
        studio.youtube_light = None
        studio.youtube_light_dot = None
        studio._sync_youtube_buttons = mock.Mock()

        studio._sync_youtube_status()

        self.assertEqual(studio.youtube_status_lbl.kwargs["text"], "YOUTUBE FAILED")

if __name__ == "__main__":
    unittest.main()

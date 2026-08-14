import io
import os
import time
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
    AUDIO_PREROLL_SECONDS,
    AUDIO_STALL_RESTART_SECONDS,
    DEMUCS_MODEL_FILENAME,
    _PcmReader,
    _audio_buffer_seconds,
    _audio_max_preroll_seconds,
    _audio_preroll_seconds,
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

    def test_ytdlp_retries_empty_format_list_with_browser_cookies(self):
        """"No video formats found" is a bot block, not a broken video.

        It used to end the run on the first cookieless attempt, so the browser
        cookies that satisfy the request were never tried and a link that plays
        fine in the browser came back as "youtube failed".
        """
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
                        "[youtube] BLr8IzA_lcs: No video formats found!; "
                        "please report this issue")
                return {"title": "OK", "download": download}

        fake_module = types.SimpleNamespace(YoutubeDL=FakeYDL)
        with mock.patch.dict(
                os.environ,
                {"AVATAR_YOUTUBE_COOKIE_BROWSERS": "edge"},
                clear=False):
            info = extract_info_with_retries(
                fake_module, "https://youtube.test/watch?v=abc",
                {"quiet": True}, download=False)

        self.assertEqual(info["title"], "OK")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1]["cookiesfrombrowser"], ("edge",))

    def test_live_voice_buffer_stays_shallow_enough_to_brake_ffmpeg(self):
        """The brake on a live voice is load bearing - do not remove it.

        It was removed once, on the theory that a braked ffmpeg drifts off the
        live edge and that this was why the voice starved 27-29 s into every
        live playback. Unbraked, ffmpeg raced through the segments the playlist
        had, ran out, and exited cleanly 25 s in: the voice stopped altogether
        rather than briefly starving. Pacing ffmpeg to the rate the audio is
        consumed is what keeps it following a live playlist, so a live buffer
        deep enough to never fill is a total loss of voice waiting to happen.
        """
        live = _audio_buffer_seconds(live=True)
        self.assertLessEqual(live, 60.0)
        self.assertEqual(live, _audio_buffer_seconds(live=False))

    def test_live_voice_cushion_is_not_tied_to_the_picture(self):
        """The voice buys its own depth; the picture corrects the difference.

        Holding the two live pre-rolls equal was tried and starved this side:
        at the picture's 4 s the buffer ran dry within a minute of a live
        broadcast, and since the picture follows this player's position, the
        starve froze the picture too. The picture measures both lags and holds
        itself back to match, so this only has to be deep enough for the sound.
        """
        from youtube_video import LIVE_PREROLL_SECONDS

        self.assertGreaterEqual(AUDIO_PREROLL_SECONDS, LIVE_PREROLL_SECONDS)
        # The worst stall measured on a live broadcast was 9.6 s.
        self.assertGreaterEqual(_audio_max_preroll_seconds(), 9.6)

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

    def test_refused_download_streams_the_voice_instead_of_losing_it(self):
        """YouTube refuses downloads far more often than it refuses playback.

        A 403 on the download killed the voice outright while the picture kept
        playing from the very same media URLs - the download path is what was
        refused, not the media. Streaming is what the live path already does.
        """
        class FakeYDL:
            def __init__(self, opts):
                self.download = bool(opts.get("outtmpl"))

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def extract_info(self, _url, download):
                if download:
                    raise RuntimeError(
                        "unable to download video data: HTTP Error 403: Forbidden")
                return {
                    "title": "Recording",
                    "duration": 600.0,
                    "formats": [{"url": "https://example.test/audio",
                                 "acodec": "opus", "vcodec": "none",
                                 "abr": 96}],
                }

        fake_module = types.SimpleNamespace(YoutubeDL=FakeYDL)
        out = {}
        statuses = []
        with mock.patch.dict(sys.modules, {"yt_dlp": fake_module}), \
                mock.patch("youtube_audio.get_cached_audio", return_value=None):
            source, title, duration, cache_hit = _resolve_audio_source(
                "https://youtube.test/watch?v=abc",
                status_callback=statuses.append, out_info=out)

        self.assertEqual(source, "https://example.test/audio")
        self.assertEqual(title, "Recording")
        # Streamed, but still a recording: its range and its end still apply.
        self.assertTrue(out["streamed"])
        self.assertFalse(out["is_live"])
        self.assertTrue([s for s in statuses if "streaming the voice" in s])

    def test_reader_reports_when_audio_last_arrived(self):
        """The only signal that a live stream has actually stopped.

        ffmpeg staying alive proves nothing: a connection can die without ever
        raising, and the playback loop had no timeout on its refill. A stream
        that stopped delivering became permanent silence - no error, no log
        line, no recovery - while the picture carried on regardless.
        """
        proc = mock.Mock()
        proc.stdout = io.BytesIO(b"")
        proc.poll.return_value = None
        reader = _PcmReader(proc, 64, 4)
        self.addCleanup(reader.close)
        time.sleep(0.25)
        # Nothing ever arrived, so the clock has been running since it started.
        self.assertGreaterEqual(
            time.monotonic() - reader.last_progress_at, 0.2)

    def test_silence_watchdog_is_shorter_than_a_viewer_will_tolerate(self):
        # Segments are a few seconds; anything longer than this without a byte
        # is a dead stream, and every second beyond it is silence on air.
        self.assertLessEqual(AUDIO_STALL_RESTART_SECONDS, 15.0)
        self.assertGreater(AUDIO_STALL_RESTART_SECONDS, 5.0)

    def test_live_audio_selection_avoids_the_widest_rendition(self):
        """Picking the fattest audio stream is what starves the voice.

        The widest rendition maximises the bytes that have to arrive on time,
        and on a slow link that is exactly what puts holes in the voice. Speech
        that is about to be pitch shifted and rebroadcast gains nothing audible
        from the top rendition.
        """
        info = {
            "formats": [
                {"url": "https://example.test/fat", "acodec": "opus",
                 "vcodec": "none", "abr": 256},
                {"url": "https://example.test/lean", "acodec": "opus",
                 "vcodec": "none", "abr": 96},
                {"url": "https://example.test/tiny", "acodec": "opus",
                 "vcodec": "none", "abr": 48},
            ],
        }
        # Best that fits under the cap - not the widest, and not the smallest.
        self.assertEqual(
            _select_live_audio_url(info), "https://example.test/lean")

    def test_live_audio_falls_back_to_the_smallest_overshoot(self):
        # Nothing fits: take the least bad rather than the widest.
        info = {
            "formats": [
                {"url": "https://example.test/huge", "acodec": "opus",
                 "vcodec": "none", "abr": 320},
                {"url": "https://example.test/big", "acodec": "opus",
                 "vcodec": "none", "abr": 160},
            ],
        }
        self.assertEqual(
            _select_live_audio_url(info), "https://example.test/big")

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
                    mock.patch("youtube_audio.enforce_budget",
                               return_value=(0, 0, [])) as trim, \
                    mock.patch.dict(sys.modules, {"yt_dlp": fake_module}):
                source, title, duration, cache_hit = _resolve_audio_source(
                    "https://youtube.test/watch?v=abc123")

            # The freshly downloaded video must be exempt from its own cleanup.
            self.assertEqual(trim.call_args.kwargs["keep_ids"], ["abc123"])
            self.assertLessEqual(
                int(ydl_opts[-1]["format"].split("abr<=?")[1].split("]")[0]), 160)

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


class AudioJitterBufferTests(unittest.TestCase):
    """A live broadcast delivers audio in bursts, not on the speaker's beat.

    Measured over 60 s of one: ffmpeg supplied 1.09x realtime overall - enough -
    but in bursts separated by 21 stalls, the longest 9.6 s. Reading a block and
    then blocking to play it turned every stall into a hole in the sound. With
    the buffer, the same stream played 150 s with zero holes at 1.000x realtime.
    """

    def _reader(self, chunks, block_bytes, max_blocks=8):
        proc = mock.Mock()
        proc.poll.return_value = None
        stdout = mock.Mock()
        remaining = list(chunks)

        def _read(size):
            if not remaining:
                return b""
            piece = remaining[0]
            head, tail = piece[:size], piece[size:]
            if tail:
                remaining[0] = tail
            else:
                remaining.pop(0)
            return head

        stdout.read.side_effect = _read
        proc.stdout = stdout
        reader = _PcmReader(proc, block_bytes, max_blocks)
        reader._thread.join(timeout=5.0)
        return reader

    def test_reader_fills_ahead_of_playback(self):
        # The point of the buffer: blocks accumulate with nobody popping.
        reader = self._reader([b"\x01\x02" * 300], block_bytes=200)

        self.assertEqual(reader.depth(), 3)
        self.assertEqual(len(reader.pop()), 200)

    def test_full_buffer_waits_instead_of_dropping_audio(self):
        # Dropping a block would be an audible skip; back-pressure is correct.
        reader = self._reader([b"\x01" * 2000], block_bytes=200, max_blocks=4)
        try:
            self.assertEqual(reader.depth(), 4)
            self.assertFalse(reader.eof, "reader must still be holding audio")
        finally:
            reader.close()

    def test_reader_reports_eof_and_drains(self):
        reader = self._reader([b"\x01" * 400], block_bytes=200)

        self.assertTrue(reader.eof)
        self.assertFalse(reader.drained())
        reader.pop()
        reader.pop()
        self.assertTrue(reader.drained())

    def test_short_final_block_is_not_played_as_a_whole_one(self):
        reader = self._reader([b"\x01" * 250], block_bytes=200)

        self.assertEqual(reader.depth(), 1)
        self.assertEqual(reader.short_tail, b"\x01" * 50)

    def test_preroll_survives_the_stalls_that_were_measured(self):
        # The worst observed stall on a live broadcast was 9.6 s.
        self.assertGreaterEqual(_audio_max_preroll_seconds(), 9.6)

    def test_buffer_is_never_shallower_than_the_preroll_it_holds(self):
        with mock.patch.dict(
                os.environ,
                {"AVATAR_YOUTUBE_AUDIO_PREROLL": "20",
                 "AVATAR_YOUTUBE_AUDIO_BUFFER": "1"}, clear=False):
            self.assertGreaterEqual(_audio_buffer_seconds(), 20.0)

    def test_preroll_is_configurable(self):
        with mock.patch.dict(
                os.environ, {"AVATAR_YOUTUBE_AUDIO_PREROLL": "3.5"}, clear=False):
            self.assertEqual(_audio_preroll_seconds(), 3.5)

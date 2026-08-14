import io
import os
import sys
import threading
import time
import types
import unittest
from unittest import mock

import numpy as np


ENGINES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "engines")
if ENGINES_DIR not in sys.path:
    sys.path.insert(0, ENGINES_DIR)

from youtube_video import (
    AUDIO_CLOCK_STALL_LIMIT,
    LOCAL_BUFFER_SECONDS,
    LIVE_MAX_PREROLL_SECONDS,
    MAX_CACHE_DURATION_SECONDS,
    VIDEO_FPS,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
    YouTubeVideoScene,
    _AudioClock,
    _DecoderStream,
    _cache_duration_limit,
    _decoder_needs_restart,
    _download_tuning_opts,
    _ffmpeg_header_text,
    _is_usable_preview_file,
    _preview_video_candidates,
    _extract_video_info,
    _select_video_source,
    _too_long_to_cache,
    _video_buffer_seconds,
    normalized_crop,
    probe_youtube_live,
    resolve_youtube_video,
)


class DecoderBackpressureTests(unittest.TestCase):
    """A recording must play every frame; only live may drop."""

    def _stream(self, frame_count, buffer_frames, block_when_full):
        frame_bytes = VIDEO_WIDTH * VIDEO_HEIGHT * 3
        data = b"".join(
            bytes([i % 256]) * frame_bytes for i in range(frame_count))

        proc = mock.Mock()
        proc.stdout = io.BytesIO(data)
        proc.poll.return_value = None
        return _DecoderStream(
            proc, buffer_frames, block_when_full=block_when_full)

    def test_recording_decoder_does_not_throw_frames_away(self):
        """The bug behind "the video is moving fast".

        ffmpeg decodes a local file far faster than realtime. With a dropping
        buffer it overran continuously and only the newest frames survived, so
        consecutive frames shown were seconds apart and the picture raced
        through the video at the right frame rate.
        """
        stream = self._stream(40, buffer_frames=5, block_when_full=True)
        self.addCleanup(stream.close)
        first = [stream.pop() for _ in range(3)]
        time.sleep(0.3)          # plenty of time to overrun a 5-frame buffer
        self.assertEqual(stream.dropped, 0)
        # Frames come out in order, none skipped.
        self.assertEqual([f[0] for f in first if f], [0, 1, 2])

    def test_a_braked_decoder_is_not_mistaken_for_a_stalled_one(self):
        # Waiting on a full buffer is health, not a stall: reading it as one
        # resynced the decoder every time the buffer filled.
        stream = self._stream(40, buffer_frames=5, block_when_full=True)
        self.addCleanup(stream.close)
        time.sleep(0.3)
        self.assertLess(stream.stalled_for(), 0.2)

    def test_local_buffer_absorbs_a_cpu_starved_decoder(self):
        """A braked decoder's only slack is this buffer.

        The studio runs face swap and restoration across most of the machine,
        and a descheduled ffmpeg was measured going quiet for 3 s mid-file with
        the decoder alive and nothing dropped. Shallower than that pause and
        the buffer empties, which the scene loop reads as a dead decoder and
        pays for with a resync and a visible jump.
        """
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AVATAR_YOUTUBE_LOCAL_BUFFER", None)
            self.assertGreaterEqual(LOCAL_BUFFER_SECONDS, 3.0)

    def test_live_decoder_still_drops_to_stay_at_the_edge(self):
        stream = self._stream(40, buffer_frames=5, block_when_full=False)
        self.addCleanup(stream.close)
        deadline = time.monotonic() + 5.0
        while stream.dropped == 0 and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertGreater(stream.dropped, 0)
        self.assertLessEqual(stream.depth(), 5)


class AudioClockTests(unittest.TestCase):
    """The reference live frames are metered against."""

    def test_clock_parked_during_voice_preroll_has_not_started(self):
        # The voice reports its start position while it fills its own cushion.
        # Treating that as a running clock would anchor the picture to it and
        # start the picture ahead of the sound - the bug this class fixes.
        clock = _AudioClock(lambda: 0.0)
        for _ in range(5):
            self.assertEqual(clock.read(), 0.0)
        self.assertFalse(clock.started())

    def test_clock_starts_once_the_position_actually_moves(self):
        positions = iter([0.0, 0.0, 0.021, 0.042])
        clock = _AudioClock(lambda: next(positions))
        clock.read()
        clock.read()
        self.assertFalse(clock.started())
        clock.read()
        self.assertTrue(clock.started())
        self.assertLess(clock.idle_for(), 1.0)

    def test_swapping_to_a_new_voice_does_not_count_as_playback(self):
        """A changed reading is not the same as sound having played.

        Starting a second link replaces the voice, so the position read jumps
        from the old player's to the new one's start. Counted as movement, that
        released the picture the instant the new ffmpeg launched - before its
        pre-roll had produced a sample - and the head start was back.
        """
        positions = iter([120.0, 0.0, 0.0, 0.021])
        clock = _AudioClock(lambda: next(positions))
        clock.read()
        clock.read()                      # 120 -> 0: a swap, not playback
        self.assertFalse(clock.started())
        clock.read()
        self.assertFalse(clock.started())
        clock.read()                      # now it actually advances
        self.assertTrue(clock.started())

    def test_a_forward_jump_is_not_playback_either(self):
        # Losing the voice hands the reading to the studio's fallback clock,
        # which is a jump forward, not sound.
        positions = iter([0.0, 0.021, 900.0, 900.0])
        clock = _AudioClock(lambda: next(positions))
        clock.read()
        clock.read()
        self.assertTrue(clock.started())
        clock.read()                      # 0.021 -> 900: a different clock
        self.assertFalse(clock.started())

    def test_frozen_clock_reports_how_long_it_has_been_idle(self):
        clock = _AudioClock(lambda: 5.0)
        clock.read()
        with mock.patch("youtube_video.time.monotonic",
                        return_value=clock._moved_at + 9.0):
            self.assertGreaterEqual(clock.idle_for(), 9.0)

    def test_unusable_clock_reads_as_none(self):
        self.assertIsNone(_AudioClock(None).read())
        self.assertIsNone(_AudioClock(lambda: None).read())
        self.assertIsNone(_AudioClock(lambda: "nope").read())

        def boom():
            raise RuntimeError("no player")

        self.assertIsNone(_AudioClock(boom).read())

    def test_video_holds_frames_long_enough_to_ride_out_a_voice_stall(self):
        """The buffer has to cover what the sync lock can hold back.

        Metering on the voice means an audio stall parks the picture, and the
        frames arriving meanwhile pile up in the jitter buffer. The buffer must
        outlast the deepest pre-roll plus the longest hold the meter allows,
        or ffmpeg jams on the pipe and falls off the broadcast edge.
        """
        with mock.patch.dict(os.environ, {}, clear=False):
            for key in ("AVATAR_YOUTUBE_LIVE_BUFFER",
                        "AVATAR_YOUTUBE_LIVE_MAX_PREROLL"):
                os.environ.pop(key, None)
            self.assertGreaterEqual(
                _video_buffer_seconds(),
                LIVE_MAX_PREROLL_SECONDS + AUDIO_CLOCK_STALL_LIMIT)


class LiveFrameMeterTests(unittest.TestCase):
    """A live picture is metered against the voice, not the wall clock."""

    def test_picture_gives_up_its_head_start_when_the_voice_starts(self):
        """The real flow: the picture is already running when the voice starts.

        Pasting a link starts the picture; pressing the voice button starts the
        sound seconds later, by which point the picture is that much further
        into the broadcast. Anchoring alone would pin that gap forever, so the
        picture has to hold still and let the sound draw level.
        """
        scene = YouTubeVideoScene(lambda: 0.0, voice_lag=lambda: 9.0)
        # Opened 3s ago having shown 15 frames (1s of broadcast): the picture
        # is 2s behind the edge against the voice's 9s, so it leads by 7s.
        opened = time.monotonic() - 3.0
        ahead = scene._picture_is_ahead_by(opened, 15, 1.0 / VIDEO_FPS)
        self.assertAlmostEqual(ahead, 7.0, delta=0.3)

    def test_no_hold_when_the_picture_is_the_side_that_is_behind(self):
        # Overdue frames already catch themselves up; holding would be backwards.
        scene = YouTubeVideoScene(lambda: 0.0, voice_lag=lambda: 1.0)
        opened = time.monotonic() - 8.0
        self.assertEqual(
            scene._picture_is_ahead_by(opened, 15, 1.0 / VIDEO_FPS), 0.0)

    def test_no_hold_without_a_measurable_voice(self):
        scene = YouTubeVideoScene(lambda: 0.0)
        opened = time.monotonic() - 8.0
        self.assertEqual(
            scene._picture_is_ahead_by(opened, 15, 1.0 / VIDEO_FPS), 0.0)

    def test_hold_never_exceeds_what_the_buffer_can_absorb(self):
        # A hold longer than the buffer jams ffmpeg on the pipe mid-hold, which
        # drops it off the broadcast edge - the failure the buffer prevents.
        scene = YouTubeVideoScene(lambda: 0.0, voice_lag=lambda: 600.0)
        opened = time.monotonic() - 1.0
        self.assertLessEqual(
            scene._picture_is_ahead_by(opened, 0, 1.0 / VIDEO_FPS),
            _video_buffer_seconds())

    def _run_live_scene(self, position_getter, voice_active):
        """Drive the scene loop in the background against a full jitter buffer."""
        frame = b"\x40" * (VIDEO_WIDTH * VIDEO_HEIGHT * 3)
        queued = [frame] * 400
        shown = {"count": 0}
        statuses = []
        scene = YouTubeVideoScene(
            position_getter, voice_active=voice_active,
            status_callback=statuses.append)
        scene.statuses = statuses
        scene._running = True

        class FakeStream:
            def pop(self):
                if not queued:
                    return None
                shown["count"] += 1
                return queued.pop(0)

            def depth(self):
                return len(queued)

            def alive(self):
                return True

            def returncode(self):
                return None

            def stalled_for(self):
                return 0.0

            def error_tail(self):
                return ""

            def close(self):
                pass

        resolved = {
            "source": "https://live.test/playlist.m3u8",
            "direct_source": "https://live.test/playlist.m3u8",
            "title": "Live", "duration": 0.0, "headers": {},
            "direct_headers": {}, "is_live": True, "thumbnail": "",
        }
        patches = [
            mock.patch("youtube_video.resolve_youtube_video",
                       return_value=resolved),
            mock.patch.object(scene, "_open_stream",
                              side_effect=lambda _p: FakeStream()),
            mock.patch.dict(
                os.environ, {"AVATAR_YOUTUBE_LIVE_PREROLL": "0.05"},
                clear=False),
            mock.patch("youtube_video._thumbnail_frame", return_value=None),
        ]
        for patch in patches:
            patch.start()
        thread = threading.Thread(target=scene._run, daemon=True)
        thread.start()

        def stop():
            scene._running = False
            thread.join(timeout=5.0)
            for patch in reversed(patches):
                patch.stop()

        self.addCleanup(stop)
        return scene, shown

    def test_first_frame_waits_for_the_voice_to_start(self):
        # The voice reports its start position while it fills its own cushion.
        # Showing the picture during that wait is what put the picture seconds
        # ahead of the sound on every open.
        clock = {"pos": 0.0}
        scene, shown = self._run_live_scene(lambda: clock["pos"], lambda: True)
        time.sleep(0.6)
        self.assertEqual(shown["count"], 0)
        self.assertFalse(scene.video_ready)

        clock["pos"] = 0.001
        deadline = time.monotonic() + 5.0
        while shown["count"] == 0 and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertGreaterEqual(shown["count"], 1)

    def test_holding_for_the_voice_still_reports_the_picture_as_ready(self):
        """Breaks the deadlock between the two waits.

        The studio holds the voice until the picture is ready, and a live
        picture holds frame 0 until the voice starts. If "ready" meant "a frame
        is on screen", neither could ever move: the picture would be waiting
        for the very voice that is waiting for it.
        """
        clock = {"pos": 0.0}
        scene, shown = self._run_live_scene(lambda: clock["pos"], lambda: True)
        deadline = time.monotonic() + 5.0
        while not scene.buffered_ready and time.monotonic() < deadline:
            time.sleep(0.02)

        self.assertTrue(scene.buffered_ready)
        self.assertFalse(scene.video_ready)
        self.assertEqual(shown["count"], 0)

        # ...and the frame lands once the voice that was released starts.
        clock["pos"] = 0.001
        deadline = time.monotonic() + 5.0
        while not scene.video_ready and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertTrue(scene.video_ready)

    def test_a_hold_does_not_re_anchor_itself_every_pass(self):
        """Holding the picture must not look like the voice jumping backwards.

        The hold works by putting the anchor ahead of the voice. Testing for a
        restarted voice against that anchor made every hold trip the check on
        the very next pass, re-anchoring hundreds of times a second and filling
        the log until the hold decayed under the threshold on its own.
        """
        clock = {"pos": 100.0}
        scene, shown = self._run_live_scene(
            lambda: clock["pos"], lambda: True)
        scene.voice_lag = lambda: 9.0
        deadline = time.monotonic() + 5.0
        while shown["count"] == 0 and time.monotonic() < deadline:
            clock["pos"] += 0.02
            time.sleep(0.02)

        holds = [s for s in scene.statuses if "holding" in s]
        # A hold may be announced once per genuine re-lock, never per loop pass.
        self.assertLess(len(holds), 5, f"re-anchor loop: {len(holds)} holds")

    def test_picture_keeps_moving_slowly_while_the_voice_is_stalled(self):
        """A stalled voice must slow the picture, not stop it.

        Gating frames on the voice position held sync exactly but froze the
        picture for as long as the voice was starved - six seconds at a time on
        a slow connection, which is the most visible artefact a video can have.
        Easing the frame rate absorbs the same error invisibly.
        """
        clock = {"pos": 0.0}
        scene, shown = self._run_live_scene(lambda: clock["pos"], lambda: True)
        clock["pos"] = 0.001
        deadline = time.monotonic() + 5.0
        while shown["count"] == 0 and time.monotonic() < deadline:
            time.sleep(0.02)

        # Voice frozen for a second. The old gate showed exactly one frame in
        # this window; rate control should keep the picture running near 0.85x.
        start = shown["count"]
        time.sleep(1.0)
        advanced = shown["count"] - start
        self.assertGreater(advanced, 5)
        # ...but slowed, not free-running at full speed.
        self.assertLess(advanced, VIDEO_FPS)

    def test_picture_does_not_run_away_from_a_stalled_voice(self):
        # Slowing is the correction; drifting off unchecked is what the meter
        # exists to prevent. Over a second of frozen voice the picture must
        # stay far closer than the ~15 frames free-running would have shown.
        clock = {"pos": 0.0}
        scene, shown = self._run_live_scene(lambda: clock["pos"], lambda: True)
        clock["pos"] = 0.001
        deadline = time.monotonic() + 5.0
        while shown["count"] == 0 and time.monotonic() < deadline:
            time.sleep(0.02)

        start = shown["count"]
        time.sleep(1.0)
        drift = (shown["count"] - start) / VIDEO_FPS
        # At the 0.85x floor a frozen second costs at most ~0.85 s of drift.
        self.assertLessEqual(drift, 1.0)

    def test_scene_without_a_voice_shows_frames_immediately(self):
        # A live scene with no voice has no clock to wait for, and must never
        # hold the picture black waiting for one that is never coming.
        scene, shown = self._run_live_scene(lambda: 0.0, lambda: False)
        deadline = time.monotonic() + 5.0
        while shown["count"] == 0 and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertGreaterEqual(shown["count"], 1)


class YouTubeVideoTests(unittest.TestCase):
    def test_video_scene_uses_smooth_preview_frame_rate(self):
        self.assertGreaterEqual(VIDEO_FPS, 14.0)
        self.assertLessEqual(VIDEO_FPS, 24.0)

    def test_frame_snapshot_exposes_new_frame_serial(self):
        scene = YouTubeVideoScene(lambda: 0.0)
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        with scene._lock:
            scene.latest_frame = frame
            scene.frame_serial += 1

        serial, snapshot = scene.frame_snapshot()

        self.assertEqual(serial, 1)
        self.assertIs(snapshot, frame)

    def test_decoder_catches_up_after_normal_network_startup(self):
        self.assertFalse(_decoder_needs_restart(16.0, 10.0))
        self.assertTrue(_decoder_needs_restart(45.0, 10.0))
        self.assertTrue(_decoder_needs_restart(8.0, 10.0))

    def test_video_source_prefers_highest_stream_at_or_below_720p(self):
        info = {
            "formats": [
                {"url": "360", "vcodec": "h264", "height": 360, "fps": 30},
                {"url": "720", "vcodec": "h264", "height": 720, "fps": 30},
                {"url": "1080", "vcodec": "h264", "height": 1080, "fps": 60},
                {"url": "1440", "vcodec": "h264", "height": 1440, "fps": 60},
                {"url": "2160", "vcodec": "h264", "height": 2160, "fps": 60},
                {"url": "4320", "vcodec": "h264", "height": 4320, "fps": 60},
            ]
        }
        self.assertEqual(_select_video_source(info), "720")

    def test_video_source_prefers_hd_video_over_lower_progressive(self):
        info = {
            "formats": [
                {"url": "dash-720", "vcodec": "avc1.64001f",
                 "acodec": "none", "height": 720, "fps": 30},
                {"url": "progressive-360", "vcodec": "avc1.42001e",
                 "acodec": "mp4a.40.2", "height": 360, "fps": 30},
            ]
        }

        self.assertEqual(_select_video_source(info), "dash-720")

    def test_normalized_crop_uses_fractional_coordinates(self):
        frame = np.arange(100 * 200 * 3, dtype=np.uint8).reshape(100, 200, 3)
        cropped = normalized_crop(frame, (0.25, 0.20, 0.75, 0.80))
        self.assertEqual(cropped.shape, (60, 100, 3))
        np.testing.assert_array_equal(cropped[0, 0], frame[20, 50])

    def test_resolver_returns_video_metadata_without_download(self):
        info = {
            "title": "Test video",
            "duration": 123,
            "http_headers": {"User-Agent": "test", "Accept": "global"},
            "formats": [
                {"url": "https://example.test/video", "vcodec": "h264",
                 "height": 720, "http_headers": {"Accept": "format"}}
            ],
        }

        class FakeYDL:
            def __init__(self, opts):
                self.opts = opts

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def extract_info(self, _url, download):
                self.download = download
                return info

        fake_module = types.SimpleNamespace(YoutubeDL=FakeYDL)
        with mock.patch.dict(sys.modules, {"yt_dlp": fake_module}), \
                mock.patch.dict(os.environ, {"AVATAR_YOUTUBE_VIDEO_CACHE": "0"}, clear=False):
            resolved = resolve_youtube_video("https://youtu.be/test")

        self.assertEqual(resolved["source"], "https://example.test/video")
        self.assertEqual(resolved["title"], "Test video")
        self.assertEqual(resolved["duration"], 123.0)
        self.assertEqual(resolved["headers"]["Accept"], "format")

    def test_resolver_uses_cached_preview_video_for_non_live_links(self):
        info = {
            "title": "Test video",
            "duration": 123,
            "is_live": False,
            "formats": [
                {"url": "https://example.test/video", "vcodec": "h264",
                 "height": 360}
            ],
        }

        class FakeYDL:
            def __init__(self, opts):
                self.opts = opts

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def extract_info(self, _url, download):
                return info

        fake_module = types.SimpleNamespace(YoutubeDL=FakeYDL)
        with mock.patch.dict(sys.modules, {"yt_dlp": fake_module}), \
                mock.patch("youtube_video.audio_dir", return_value="cache-dir"), \
                mock.patch("youtube_video._preview_video_candidates",
                           return_value=["cache-dir/preview.mp4"]):
            resolved = resolve_youtube_video("https://youtu.be/test")

        self.assertEqual(resolved["source"], "cache-dir/preview.mp4")
        self.assertEqual(resolved["direct_source"], "https://example.test/video")
        self.assertEqual(resolved["headers"], {})
        self.assertEqual(resolved["direct_headers"], {})

    def test_resolver_can_force_refresh_existing_preview_cache(self):
        info = {
            "title": "Test video",
            "duration": 123,
            "is_live": False,
            "formats": [
                {"url": "https://example.test/video", "vcodec": "h264",
                 "height": 360}
            ],
        }

        class FakeYDL:
            def __init__(self, opts):
                self.opts = opts

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def extract_info(self, _url, download):
                return info

        fake_module = types.SimpleNamespace(YoutubeDL=FakeYDL)
        with mock.patch.dict(sys.modules, {"yt_dlp": fake_module}), \
                mock.patch(
                    "youtube_video._cached_or_download_preview_video",
                    return_value="cache-dir/preview.mp4") as cached:
            resolved = resolve_youtube_video(
                "https://youtu.be/test", force_refresh_cache=True)

        self.assertEqual(resolved["source"], "cache-dir/preview.mp4")
        self.assertTrue(cached.call_args.kwargs["force_refresh"])

    def test_preview_cache_ignores_old_low_res_file_when_hd_missing(self):
        with mock.patch("youtube_video.glob.glob") as glob_mock, \
                mock.patch("youtube_video.os.path.isfile", return_value=True):
            glob_mock.side_effect = lambda pattern: (
                ["cache/preview-low.mp4"] if "preview-low" in pattern else [])

            self.assertEqual(_preview_video_candidates("cache"), [])

    def test_preview_cache_ignores_quarantined_or_partial_files(self):
        with mock.patch("youtube_video.os.path.isfile", return_value=True):
            self.assertFalse(_is_usable_preview_file("cache/preview-hd.mp4.part"))
            self.assertFalse(_is_usable_preview_file("cache/preview-hd.mp4.bad-1"))
            self.assertFalse(_is_usable_preview_file("cache/preview-hd.mp4.old-1"))
            self.assertFalse(_is_usable_preview_file("cache/preview-hd.tmp"))
            self.assertTrue(_is_usable_preview_file("cache/preview-hd.mp4"))

    def test_studio_announces_seen_youtube_link(self):
        from avatar_studio import AvatarStudio, AMBER

        studio = AvatarStudio.__new__(AvatarStudio)
        studio._youtube_link_cache_state = mock.Mock(
            return_value=(True, "abc123"))
        studio._set_youtube_progress = mock.Mock()
        studio._log_msg = mock.Mock()
        studio._topdraw = mock.Mock()
        studio.root = types.SimpleNamespace(after=lambda *_args: None)

        self.assertTrue(
            studio._announce_youtube_link_state("https://youtu.be/abc123"))

        self.assertEqual(
            studio._youtube_link_notice,
            "LINK WAS HERE BEFORE - USING CACHE")
        self.assertEqual(studio._youtube_link_notice_color, AMBER)
        studio._set_youtube_progress.assert_called_with(
            3, "Link was here before - using cached video")

    def test_studio_maps_video_download_progress_to_youtube_meter(self):
        from avatar_studio import AvatarStudio

        studio = AvatarStudio.__new__(AvatarStudio)
        calls = []
        studio._set_youtube_progress = lambda value, text=None: calls.append((value, text))
        studio._bump_youtube_progress = lambda value, text=None: calls.append((value, text))
        studio._youtube_progress_value = 0.0

        studio._on_youtube_video_status("video preview download 50%")

        self.assertEqual(calls[-1], (50.0, "Video downloading 50%"))

    def test_studio_waits_until_video_scene_has_first_frame(self):
        from avatar_studio import AvatarStudio

        studio = AvatarStudio.__new__(AvatarStudio)
        studio._youtube_scene = types.SimpleNamespace(
            url="https://youtu.be/test", video_ready=True, status="")
        studio._set_youtube_progress = mock.Mock()
        studio._log_msg = mock.Mock()
        studio._youtube_progress_value = 0.0

        self.assertTrue(
            studio._wait_for_youtube_video_ready(
                "https://youtu.be/test", timeout=0.01))

    def test_studio_video_and_audio_use_the_same_position_clock(self):
        from avatar_studio import AvatarStudio

        studio = AvatarStudio.__new__(AvatarStudio)
        studio._youtube_audio = types.SimpleNamespace(position_seconds=42.75)
        studio._youtube_start_seconds = 0.0
        studio._youtube_chunks = []

        self.assertEqual(studio._youtube_position_seconds(), 42.75)

    def test_caption_clock_uses_saved_playback_position_when_audio_is_absent(self):
        from avatar_studio import AvatarStudio

        studio = AvatarStudio.__new__(AvatarStudio)
        studio._youtube_audio = None
        studio._youtube_start_seconds = 10.0
        studio._youtube_end_seconds = None
        studio._youtube_duration = 0.0
        studio._youtube_chunks = ["a", "b", "c", "d"]
        studio._youtube_index = 2
        studio._youtube_scene = types.SimpleNamespace(duration=110.0)
        studio._youtube_clock_position = 18.0
        studio._youtube_clock_anchor_t = None

        self.assertEqual(studio._youtube_position_seconds(), 18.0)

    def test_scene_video_clock_stays_paused_without_anchor(self):
        from avatar_studio import AvatarStudio

        studio = AvatarStudio.__new__(AvatarStudio)
        studio._youtube_audio = None
        studio._youtube_chunks = []
        studio._youtube_start_seconds = 5.0
        studio._youtube_scene = types.SimpleNamespace(duration=100.0)
        studio._scene_source = "youtube"
        studio._youtube_clock_position = 5.0
        studio._youtube_clock_anchor_t = None

        with mock.patch("avatar_studio.time.monotonic", return_value=12.5):
            self.assertEqual(studio._youtube_position_seconds(), 5.0)

    def test_scene_video_clock_advances_from_anchor_when_playing(self):
        from avatar_studio import AvatarStudio

        studio = AvatarStudio.__new__(AvatarStudio)
        studio._youtube_audio = None
        studio._youtube_chunks = []
        studio._youtube_start_seconds = 5.0
        studio._youtube_end_seconds = None
        studio._youtube_duration = 0.0
        studio._youtube_scene = types.SimpleNamespace(duration=100.0)
        studio._scene_source = "youtube"
        studio._youtube_clock_position = 5.0
        studio._youtube_clock_anchor_t = 10.0

        with mock.patch("avatar_studio.time.monotonic", return_value=12.5):
            self.assertEqual(studio._youtube_position_seconds(), 7.5)

    def test_decoder_uses_hybrid_seek_and_safe_user_agent(self):
        scene = YouTubeVideoScene(lambda: 45.0)
        scene._source = "https://example.test/video.mp4"
        scene._headers = {
            "User-Agent": "browser",
            "Accept": "*/*",
            "Referer": "https://www.youtube.com/",
            "Sec-Fetch-Mode": "navigate",
        }
        fake_proc = mock.Mock()
        with mock.patch("youtube_video.subprocess.Popen",
                        return_value=fake_proc) as popen:
            self.assertIs(scene._open_decoder(45.0), fake_proc)

        cmd = popen.call_args.args[0]
        seek_indexes = [i for i, part in enumerate(cmd) if part == "-ss"]
        self.assertEqual(len(seek_indexes), 2)
        self.assertLess(seek_indexes[0], cmd.index("-i"))
        self.assertGreater(seek_indexes[1], cmd.index("-i"))
        self.assertEqual(cmd[seek_indexes[0] + 1], "40.0")
        self.assertEqual(cmd[seek_indexes[1] + 1], "5.0")
        self.assertEqual(cmd[cmd.index("-user_agent") + 1], "browser")
        header_text = cmd[cmd.index("-headers") + 1]
        self.assertIn("Accept: */*", header_text)
        self.assertIn("Referer: https://www.youtube.com/", header_text)
        self.assertNotIn("Sec-Fetch-Mode", header_text)
        vf = cmd[cmd.index("-vf") + 1]
        self.assertIn("scale=1280:720", vf)
        self.assertIn("force_original_aspect_ratio=decrease", vf)
        self.assertIn("pad=1280:720", vf)

    def test_live_decoder_never_seeks_and_never_reconnects_at_eof(self):
        # -reconnect_at_eof makes ffmpeg loop forever on the ordinary segment EOF
        # of a live HLS playlist, so the scene never receives a single frame.
        scene = YouTubeVideoScene(lambda: 620.0)
        scene.is_live = True
        scene._source = "https://manifest.googlevideo.test/hls_playlist/x.m3u8"
        scene._headers = {"User-Agent": "browser"}
        fake_proc = mock.Mock()
        with mock.patch("youtube_video.subprocess.Popen",
                        return_value=fake_proc) as popen:
            self.assertIs(scene._open_decoder(620.0), fake_proc)

        cmd = popen.call_args.args[0]
        self.assertNotIn("-ss", cmd)
        self.assertNotIn("-reconnect_at_eof", cmd)
        self.assertIn("-reconnect", cmd)
        self.assertEqual(cmd[cmd.index("-rw_timeout") + 1], "15000000")

    def test_recorded_video_still_reconnects_at_eof(self):
        scene = YouTubeVideoScene(lambda: 45.0)
        scene.is_live = False
        scene._source = "https://example.test/video.mp4"
        scene._headers = {}
        fake_proc = mock.Mock()
        with mock.patch("youtube_video.subprocess.Popen",
                        return_value=fake_proc) as popen:
            scene._open_decoder(45.0)

        cmd = popen.call_args.args[0]
        self.assertIn("-reconnect_at_eof", cmd)
        self.assertIn("-ss", cmd)

    def test_live_video_prefers_the_rolling_hls_playlist(self):
        info = {
            "is_live": True,
            "formats": [
                {"url": "https://example.test/segment.mp4", "vcodec": "avc1",
                 "ext": "mp4", "protocol": "https", "height": 720},
                {"url": "https://example.test/live.m3u8", "vcodec": "avc1",
                 "ext": "mp4", "protocol": "m3u8_native", "height": 720},
            ],
        }

        self.assertEqual(
            _select_video_source(info), "https://example.test/live.m3u8")

    def test_recorded_video_still_prefers_the_direct_https_file(self):
        info = {
            "is_live": False,
            "formats": [
                {"url": "https://example.test/live.m3u8", "vcodec": "avc1",
                 "ext": "mp4", "protocol": "m3u8_native", "height": 720},
                {"url": "https://example.test/video.mp4", "vcodec": "avc1",
                 "ext": "mp4", "protocol": "https", "height": 720},
            ],
        }

        self.assertEqual(
            _select_video_source(info), "https://example.test/video.mp4")

    def test_live_scene_refreshes_the_expiring_playlist_url(self):
        scene = YouTubeVideoScene(lambda: 0.0)
        scene.is_live = True
        scene.url = "https://youtu.be/livetest"
        scene._live_resolved_at = 0.0
        scene._source = "https://old.test/live.m3u8"

        with mock.patch("youtube_video.time.monotonic", return_value=1000.0), \
                mock.patch("youtube_video.resolve_youtube_video",
                           return_value={
                               "source": "https://fresh.test/live.m3u8",
                               "direct_source": "https://fresh.test/live.m3u8",
                               "headers": {}, "direct_headers": {},
                               "is_live": True,
                           }) as resolve:
            self.assertTrue(scene._refresh_live_source())

        resolve.assert_called_once_with("https://youtu.be/livetest")
        self.assertEqual(scene._source, "https://fresh.test/live.m3u8")

    def test_live_playlist_refresh_waits_out_its_cooldown(self):
        scene = YouTubeVideoScene(lambda: 0.0)
        scene.is_live = True
        scene.url = "https://youtu.be/livetest"

        with mock.patch("youtube_video.time.monotonic", return_value=1.0):
            scene._live_resolved_at = 0.5
            with mock.patch("youtube_video.resolve_youtube_video") as resolve:
                self.assertFalse(scene._refresh_live_source())
            resolve.assert_not_called()

    def test_studio_sends_a_live_link_to_the_real_live_voice(self):
        from avatar_studio import AvatarStudio

        studio = AvatarStudio.__new__(AvatarStudio)
        studio._youtube_scene = types.SimpleNamespace(
            url="https://youtu.be/live", video_ready=True,
            status="live video scene ready", is_live=True)

        self.assertTrue(
            studio._youtube_scene_live_state(
                "https://youtu.be/live", timeout=0.05))

    def test_studio_treats_a_recorded_link_as_captionable(self):
        from avatar_studio import AvatarStudio

        studio = AvatarStudio.__new__(AvatarStudio)
        studio._youtube_scene = types.SimpleNamespace(
            url="https://youtu.be/vod", video_ready=True,
            status="video scene ready", is_live=False)

        self.assertFalse(
            studio._youtube_scene_live_state(
                "https://youtu.be/vod", timeout=0.05))

    def test_resolver_moves_to_the_next_client_when_one_returns_no_formats(self):
        # The real failure: on 2026-08-13 a live link resolved fine one minute
        # and raised "No video formats found!" the next, because a single
        # player client was pinned. Every other client was healthy at the time.
        live_info = {
            "is_live": True,
            "title": "live",
            "formats": [
                {"url": "https://example.test/live.m3u8", "vcodec": "avc1",
                 "height": 720, "protocol": "m3u8_native"},
            ],
        }
        clients_tried = []

        def _extract(_yt_dlp, _url, opts, **_kwargs):
            clients = (opts.get("extractor_args", {})
                       .get("youtube", {}).get("player_client"))
            clients_tried.append(tuple(clients) if clients else None)
            if len(clients_tried) == 1:
                raise RuntimeError(
                    "ERROR: [youtube] abc: No video formats found!; please "
                    "report this issue")
            return live_info

        with mock.patch("youtube_video.extract_info_with_retries", _extract):
            info = _extract_video_info(object(), "https://youtu.be/abc", {})

        self.assertIs(info, live_info)
        self.assertEqual(len(clients_tried), 2)
        self.assertNotEqual(clients_tried[0], clients_tried[1])

    def test_resolver_moves_on_when_a_client_answers_with_nothing_playable(self):
        empty = {"is_live": True, "formats": []}
        good = {
            "is_live": True,
            "formats": [
                {"url": "https://example.test/live.m3u8", "vcodec": "avc1",
                 "height": 720, "protocol": "m3u8_native"},
            ],
        }
        answers = [empty, good]

        with mock.patch("youtube_video.extract_info_with_retries",
                        side_effect=lambda *a, **k: answers.pop(0)):
            info = _extract_video_info(object(), "https://youtu.be/abc", {})

        self.assertIs(info, good)

    def test_resolver_does_not_walk_the_ladder_for_a_real_failure(self):
        # A private video, a sign-in wall or a dead network answers the same
        # way from every client, so retrying is only latency on a lost cause.
        calls = []

        def _extract(*_args, **_kwargs):
            calls.append(1)
            raise RuntimeError("ERROR: [youtube] abc: Video unavailable")

        with mock.patch("youtube_video.extract_info_with_retries", _extract):
            with self.assertRaises(RuntimeError):
                _extract_video_info(object(), "https://youtu.be/abc", {})

        self.assertEqual(len(calls), 1)

    def test_resolver_reports_the_first_real_error_when_every_client_fails(self):
        with mock.patch(
                "youtube_video.extract_info_with_retries",
                side_effect=RuntimeError("No video formats found!")):
            with self.assertRaises(RuntimeError) as caught:
                _extract_video_info(object(), "https://youtu.be/abc", {})
        self.assertIn("No video formats found", str(caught.exception))

    def test_live_probe_ignores_format_filters_and_pinned_clients(self):
        # The probe exists to answer "is this live" when format selection is
        # exactly what is broken, so it must not depend on either.
        seen = {}

        def _extract(_yt_dlp, _url, opts, **_kwargs):
            seen.update(opts)
            return {"live_status": "is_live"}

        with mock.patch.dict(sys.modules, {"yt_dlp": types.SimpleNamespace()}):
            with mock.patch("youtube_video.extract_info_with_retries", _extract):
                self.assertTrue(probe_youtube_live("https://youtu.be/live"))

        self.assertNotIn("format", seen)
        self.assertNotIn("extractor_args", seen)

    def test_live_probe_returns_none_when_youtube_cannot_be_reached(self):
        with mock.patch.dict(sys.modules, {"yt_dlp": types.SimpleNamespace()}):
            with mock.patch("youtube_video.extract_info_with_retries",
                            side_effect=RuntimeError("network down")):
                self.assertIsNone(probe_youtube_live("https://youtu.be/live"))

    def test_studio_asks_youtube_when_the_scene_cannot_say_if_a_link_is_live(self):
        from avatar_studio import AvatarStudio

        studio = AvatarStudio.__new__(AvatarStudio)
        # The scene failed to resolve, so it has no answer to give.
        studio._youtube_scene = types.SimpleNamespace(
            url="https://youtu.be/live", video_ready=False,
            status="video scene failed: no video formats found", is_live=False)
        studio._log_msg = lambda *_args, **_kwargs: None

        with mock.patch("youtube_video.probe_youtube_live", return_value=True):
            self.assertTrue(
                studio._youtube_link_is_live("https://youtu.be/live"))

    def test_studio_trusts_the_scene_when_it_does_have_an_answer(self):
        from avatar_studio import AvatarStudio

        studio = AvatarStudio.__new__(AvatarStudio)
        studio._youtube_scene = types.SimpleNamespace(
            url="https://youtu.be/vod", video_ready=True,
            status="video scene ready", is_live=False)
        studio._log_msg = lambda *_args, **_kwargs: None

        with mock.patch("youtube_video.probe_youtube_live") as probe:
            self.assertFalse(
                studio._youtube_link_is_live("https://youtu.be/vod"))
        probe.assert_not_called()

    def test_decoder_does_not_use_network_reconnect_for_cached_file(self):
        scene = YouTubeVideoScene(lambda: 45.0)
        scene._source = r"C:\cache\preview-low.mp4"
        scene._headers = {}
        fake_proc = mock.Mock()
        with mock.patch("youtube_video.subprocess.Popen",
                        return_value=fake_proc) as popen:
            self.assertIs(scene._open_decoder(45.0), fake_proc)

        cmd = popen.call_args.args[0]
        self.assertNotIn("-reconnect", cmd)
        self.assertNotIn("-rw_timeout", cmd)

    def _live_scene_with_stream(self, frames, alive=True, stalled_for=0.0,
                                returncode=None, max_frames=None):
        """Run the scene loop against a scripted jitter buffer."""
        statuses = []
        scene = YouTubeVideoScene(lambda: 0.0, status_callback=statuses.append)
        scene.statuses = statuses
        scene._running = True
        opened = []
        queued = list(frames)
        shown = {"count": 0}

        class FakeStream:
            def pop(self):
                if not queued:
                    return None
                shown["count"] += 1
                frame = queued.pop(0)
                if max_frames is not None and shown["count"] >= max_frames:
                    scene._running = False
                return frame

            def depth(self):
                return len(queued)

            def alive(self):
                return alive

            def returncode(self):
                return returncode

            def stalled_for(self):
                return stalled_for

            def error_tail(self):
                return ""

            def close(self):
                pass

        def _open(_position):
            opened.append(1)
            if len(opened) > 3:
                scene._running = False
            return FakeStream()

        resolved = {
            "source": "https://live.test/playlist.m3u8",
            "direct_source": "https://live.test/playlist.m3u8",
            "title": "Live", "duration": 0.0, "headers": {},
            "direct_headers": {}, "is_live": True, "thumbnail": "",
        }
        with mock.patch("youtube_video.resolve_youtube_video",
                        return_value=resolved), \
                mock.patch.object(scene, "_open_stream", side_effect=_open), \
                mock.patch.dict(
                    os.environ, {"AVATAR_YOUTUBE_LIVE_PREROLL": "0.05"},
                    clear=False), \
                mock.patch("youtube_video._thumbnail_frame", return_value=None):
            scene._run()
        return scene, len(opened)

    def test_live_decoder_survives_an_empty_buffer_while_it_refills(self):
        # ffmpeg goes quiet for a few seconds while it reconnects a dropped TLS
        # connection. That must hold the picture and refill, not restart a
        # decoder that is still alive.
        frame = b"\x40" * (VIDEO_WIDTH * VIDEO_HEIGHT * 3)
        scene, opened = self._live_scene_with_stream(
            [frame, frame], alive=True, stalled_for=0.5, max_frames=2)

        self.assertEqual(opened, 1)          # decoder was never restarted
        self.assertTrue(scene.video_ready)
        self.assertEqual(scene.latest_frame.shape,
                         (VIDEO_HEIGHT, VIDEO_WIDTH, 3))
        self.assertFalse([s for s in scene.statuses if "resyncing" in s],
                         scene.statuses)

    def test_replacing_a_scene_is_not_reported_as_a_stream_failure(self):
        statuses = []
        scene = YouTubeVideoScene(lambda: 0.0, status_callback=statuses.append)
        scene._running = True

        class DeadStream:
            def pop(self):
                # stop() lands while the buffer is empty, exactly as it does
                # when a new link replaces the scene.
                scene._running = False
                return None

            depth = staticmethod(lambda: 0)
            alive = staticmethod(lambda: False)
            returncode = staticmethod(lambda: 1)
            stalled_for = staticmethod(lambda: 99.0)
            error_tail = staticmethod(lambda: "")
            close = staticmethod(lambda: None)

        resolved = {
            "source": "https://live.test/playlist.m3u8",
            "direct_source": "https://live.test/playlist.m3u8",
            "title": "Live", "duration": 0.0, "headers": {},
            "direct_headers": {}, "is_live": True, "thumbnail": "",
        }
        with mock.patch("youtube_video.resolve_youtube_video",
                        return_value=resolved), \
                mock.patch.object(scene, "_open_stream",
                                  return_value=DeadStream()), \
                mock.patch("youtube_video._thumbnail_frame", return_value=None):
            scene._run()

        self.assertFalse([s for s in statuses if "hiccup" in s or "resyncing" in s],
                         statuses)

    def test_live_decoder_restarts_when_the_process_actually_died(self):
        scene, opened = self._live_scene_with_stream(
            [], alive=False, stalled_for=99.0, returncode=1)

        self.assertGreaterEqual(opened, 2)   # a dead decoder is reopened
        self.assertFalse(scene.video_ready)
        self.assertTrue(any("hiccup" in s for s in scene.statuses), scene.statuses)

    def test_finished_video_is_not_mistaken_for_a_broken_download(self):
        scene = YouTubeVideoScene(lambda: 0.0)
        scene.duration = 19.0

        self.assertFalse(scene._at_end_of_video(10.0))
        self.assertTrue(scene._at_end_of_video(18.5))
        self.assertTrue(scene._at_end_of_video(25.0))
        # A live broadcast and an unknown duration both have no "end" to hit.
        self.assertFalse(scene._at_end_of_video(25.0, live=True))
        scene.duration = 0.0
        self.assertFalse(scene._at_end_of_video(25.0))

    def test_decoder_falls_back_to_direct_stream_when_cached_file_is_bad(self):
        scene = YouTubeVideoScene(lambda: 0.0)
        scene._source = r"C:\cache\preview-hd.mp4"
        scene._direct_source = "https://example.test/video.mp4"
        scene._direct_headers = {"Accept": "video/*"}

        with mock.patch("youtube_video.os.path.isfile", return_value=True), \
                mock.patch("youtube_video.os.remove") as remove:
            self.assertTrue(scene._fallback_from_bad_cache())

        remove.assert_called_once_with(r"C:\cache\preview-hd.mp4")
        self.assertEqual(scene._source, "https://example.test/video.mp4")
        self.assertEqual(scene._headers, {"Accept": "video/*"})

    def test_resolver_streams_at_once_and_caches_in_the_background(self):
        info = {
            "title": "Test video",
            "duration": 123,
            "is_live": False,
            "formats": [
                {"url": "https://example.test/video", "vcodec": "h264",
                 "height": 720}
            ],
        }

        class FakeYDL:
            def __init__(self, opts):
                self.opts = opts

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def extract_info(self, _url, download):
                return info

        fake_module = types.SimpleNamespace(YoutubeDL=FakeYDL)
        with mock.patch.dict(sys.modules, {"yt_dlp": fake_module}), \
                mock.patch("youtube_video.audio_dir", return_value="cache-dir"), \
                mock.patch("youtube_video._preview_video_candidates",
                           return_value=[]), \
                mock.patch(
                    "youtube_video._start_background_preview_download") as bg:
            resolved = resolve_youtube_video("https://youtu.be/test")

        # The first frame must not wait for the whole 720p file to land.
        self.assertEqual(resolved["source"], "https://example.test/video")
        bg.assert_called_once()

    def test_resolver_waits_for_cache_when_preloading(self):
        info = {
            "title": "Test video",
            "duration": 123,
            "is_live": False,
            "formats": [
                {"url": "https://example.test/video", "vcodec": "h264",
                 "height": 720}
            ],
        }

        class FakeYDL:
            def __init__(self, opts):
                self.opts = opts

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def extract_info(self, _url, download):
                return info

        fake_module = types.SimpleNamespace(YoutubeDL=FakeYDL)
        with mock.patch.dict(sys.modules, {"yt_dlp": fake_module}), \
                mock.patch("youtube_video.audio_dir", return_value="cache-dir"), \
                mock.patch("youtube_video._preview_video_candidates",
                           return_value=[]), \
                mock.patch("youtube_video._cached_or_download_preview_video",
                           return_value="cache-dir/preview-hd.mp4") as cached, \
                mock.patch(
                    "youtube_video._start_background_preview_download") as bg:
            resolved = resolve_youtube_video(
                "https://youtu.be/test", wait_for_cache=True)

        self.assertEqual(resolved["source"], "cache-dir/preview-hd.mp4")
        cached.assert_called_once()
        bg.assert_not_called()

    def test_download_tuning_uses_big_chunks_and_parallel_fragments(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AVATAR_YOUTUBE_DL_CHUNK_MB", None)
            os.environ.pop("AVATAR_YOUTUBE_DL_THREADS", None)
            opts = _download_tuning_opts()

        # 1 MB chunks measured 1.5 MB/s against 2.8 MB/s for this config.
        self.assertGreaterEqual(opts["http_chunk_size"], 8 * 1024 * 1024)
        self.assertGreaterEqual(opts["concurrent_fragment_downloads"], 4)

    def test_download_chunking_can_be_disabled_by_env(self):
        with mock.patch.dict(
                os.environ, {"AVATAR_YOUTUBE_DL_CHUNK_MB": "0"}, clear=False):
            opts = _download_tuning_opts()
        self.assertNotIn("http_chunk_size", opts)

    def test_decoder_moves_onto_cached_file_once_download_finishes(self):
        scene = YouTubeVideoScene(lambda: 0.0)
        scene.url = "https://youtu.be/test"
        scene._source = "https://example.test/video.mp4"
        scene._headers = {"Accept": "video/*"}

        with mock.patch("youtube_video._existing_preview_video",
                        return_value=r"C:\cache\preview-hd.mp4"):
            self.assertTrue(scene._adopt_cached_source_if_ready())

        self.assertEqual(scene._source, r"C:\cache\preview-hd.mp4")
        self.assertEqual(scene._headers, {})

    def test_decoder_never_readopts_a_cache_it_judged_broken(self):
        scene = YouTubeVideoScene(lambda: 0.0)
        scene.url = "https://youtu.be/test"
        scene._source = r"C:\cache\preview-hd.mp4"
        scene._direct_source = "https://example.test/video.mp4"

        with mock.patch("youtube_video.os.path.isfile", return_value=True), \
                mock.patch("youtube_video.os.remove"):
            self.assertTrue(scene._fallback_from_bad_cache())

        with mock.patch("youtube_video._existing_preview_video",
                        return_value=r"C:\cache\preview-hd.mp4"):
            self.assertFalse(scene._adopt_cached_source_if_ready())
        self.assertEqual(scene._source, "https://example.test/video.mp4")

    def test_expired_stream_url_is_refreshed_not_retried_forever(self):
        """A stream-first scene plays a googlevideo URL, and those expire."""
        scene = YouTubeVideoScene(lambda: 0.0)
        scene.url = "https://youtu.be/test"
        scene._source = "https://example.test/expired.mp4"

        fresh = {
            "source": "https://example.test/fresh.mp4",
            "direct_source": "https://example.test/fresh.mp4",
            "direct_headers": {"Accept": "video/*"},
            "title": "t", "duration": 1.0, "headers": {}, "is_live": False,
        }
        with mock.patch("youtube_video._existing_preview_video", return_value=""), \
                mock.patch("youtube_video.resolve_youtube_video",
                           return_value=fresh) as resolve:
            self.assertTrue(scene._refresh_direct_source())

        resolve.assert_called_once()
        self.assertEqual(scene._source, "https://example.test/fresh.mp4")

    def test_stream_refresh_prefers_the_cached_file_when_it_has_landed(self):
        scene = YouTubeVideoScene(lambda: 0.0)
        scene.url = "https://youtu.be/test"
        scene._source = "https://example.test/expired.mp4"

        with mock.patch("youtube_video._existing_preview_video",
                        return_value=r"C:\cache\preview-hd.mp4"), \
                mock.patch("youtube_video.resolve_youtube_video") as resolve:
            self.assertTrue(scene._refresh_direct_source())

        # No point re-resolving a URL when the local file is already there.
        resolve.assert_not_called()
        self.assertEqual(scene._source, r"C:\cache\preview-hd.mp4")

    def test_live_scene_does_not_use_the_vod_stream_refresh(self):
        scene = YouTubeVideoScene(lambda: 0.0)
        scene.url = "https://youtu.be/test"
        scene.is_live = True
        with mock.patch("youtube_video.resolve_youtube_video") as resolve:
            self.assertFalse(scene._refresh_direct_source())
        resolve.assert_not_called()

    def test_live_scene_never_swaps_to_a_cached_file(self):
        scene = YouTubeVideoScene(lambda: 0.0)
        scene.url = "https://youtu.be/test"
        scene.is_live = True
        scene._source = "https://example.test/live.m3u8"

        with mock.patch("youtube_video._existing_preview_video",
                        return_value=r"C:\cache\preview-hd.mp4"):
            self.assertFalse(scene._adopt_cached_source_if_ready())
        self.assertEqual(scene._source, "https://example.test/live.m3u8")

    def test_decoder_fallback_refuses_when_no_direct_stream_exists(self):
        scene = YouTubeVideoScene(lambda: 0.0)
        scene._source = r"C:\cache\preview-hd.mp4"
        scene._direct_source = r"C:\cache\preview-hd.mp4"

        self.assertFalse(scene._fallback_from_bad_cache())
        self.assertEqual(scene._source, r"C:\cache\preview-hd.mp4")

    def test_decoder_failure_keeps_last_visible_frame(self):
        scene = YouTubeVideoScene(lambda: 0.0)
        last_frame = np.full((2, 2, 3), 123, dtype=np.uint8)
        with scene._lock:
            scene.latest_frame = last_frame
            scene.frame_serial = 7

        with mock.patch("youtube_video.resolve_youtube_video",
                        side_effect=RuntimeError("network down")):
            scene._run()

        self.assertIs(scene.latest_frame, last_frame)
        self.assertEqual(scene.frame_serial, 7)
        self.assertIn("network down", scene.last_error)

    def test_ffmpeg_header_text_sanitizes_selected_headers(self):
        text = _ffmpeg_header_text({
            "Accept": "video/*",
            "Cookie": "a=b\nbad",
            "User-Agent": "separate",
            "Sec-Fetch-Mode": "navigate",
        })

        self.assertIn("Accept: video/*\r\n", text)
        self.assertIn("Cookie: a=b bad\r\n", text)
        self.assertNotIn("User-Agent", text)
        self.assertNotIn("Sec-Fetch-Mode", text)

    def test_add_scene_prefers_current_youtube_url(self):
        from avatar_studio import AvatarStudio

        class FakeEntry:
            def get(self, *_args):
                return (
                    "https://youtu.be/Wpj5TYGw4cY\n"
                    "https://youtu.be/SECONDVIDEO\n"
                )

        studio = AvatarStudio.__new__(AvatarStudio)
        studio.youtube_entry = FakeEntry()
        studio._attach_youtube_scene = mock.Mock()
        studio._start_scene_snip = mock.Mock()

        studio._add_scene()

        studio._attach_youtube_scene.assert_called_once_with(
            "https://youtu.be/Wpj5TYGw4cY", force=True)
        studio._start_scene_snip.assert_not_called()

    def test_youtube_playlist_parser_caps_at_ten_unique_links(self):
        from avatar_studio import AvatarStudio

        class FakeEntry:
            def get(self, *_args):
                return "\n".join(
                    ["not a link"]
                    + [f"https://youtu.be/video{i}" for i in range(12)]
                    + ["https://youtu.be/video3"]
                )

        studio = AvatarStudio.__new__(AvatarStudio)
        studio.youtube_entry = FakeEntry()

        urls = studio._youtube_links_from_entry()

        self.assertEqual(len(urls), 10)
        self.assertEqual(urls[0], "https://youtu.be/video0")
        self.assertEqual(urls[-1], "https://youtu.be/video9")

    def test_youtube_playlist_parser_reads_separate_link_boxes(self):
        from avatar_studio import AvatarStudio

        class FakeEntry:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

        studio = AvatarStudio.__new__(AvatarStudio)
        studio.youtube_entries = [
            FakeEntry("https://youtu.be/first"),
            FakeEntry(""),
            FakeEntry("https://youtu.be/second"),
        ]

        self.assertEqual(studio._youtube_links_from_entry(), [
            "https://youtu.be/first",
            "https://youtu.be/second",
        ])

    def test_youtube_next_video_moves_queue_index(self):
        from avatar_studio import AvatarStudio

        studio = AvatarStudio.__new__(AvatarStudio)
        studio._youtube_queue_urls = [
            "https://youtu.be/first",
            "https://youtu.be/second",
        ]
        studio._youtube_queue_slots = [0, 1]
        studio._youtube_queue_index = 0
        studio._youtube_audio = None
        studio._youtube_busy = False
        studio.running = False
        studio.engines = None
        studio._youtube_link_rows = mock.Mock(return_value=[
            (0, "https://youtu.be/first"),
            (1, "https://youtu.be/second"),
        ])
        studio._set_youtube_slot_status = mock.Mock()
        studio._youtube_clock_reset = mock.Mock()
        studio._set_youtube_progress = mock.Mock()
        studio._log_msg = mock.Mock()
        studio._attach_youtube_scene = mock.Mock()
        studio._sync_youtube_status = mock.Mock()

        self.assertTrue(studio._jump_youtube_queue(1))

        self.assertEqual(studio._youtube_queue_index, 1)
        studio._attach_youtube_scene.assert_called_once_with(
            "https://youtu.be/second", force=True, preserve_crop=True)

    def test_youtube_link_change_prepares_queue_and_waits_to_prewarm(self):
        from avatar_studio import AvatarStudio

        studio = AvatarStudio.__new__(AvatarStudio)
        studio._youtube_primary_url = mock.Mock(return_value="https://youtu.be/first")
        studio._prepare_youtube_queue = mock.Mock()
        studio._attach_youtube_scene = mock.Mock()
        studio._prewarm_youtube_queue_after_current_ready = mock.Mock()

        studio._attach_youtube_scene_from_entry()

        studio._prepare_youtube_queue.assert_called_once_with(
            "https://youtu.be/first", auto_prewarm=False)
        studio._attach_youtube_scene.assert_called_once_with(
            "https://youtu.be/first")
        studio._prewarm_youtube_queue_after_current_ready.assert_called_once_with()

    def test_youtube_queue_prepare_keeps_prewarm_state_for_same_links(self):
        from avatar_studio import AvatarStudio

        studio = AvatarStudio.__new__(AvatarStudio)
        studio._youtube_link_rows = mock.Mock(return_value=[
            (0, "https://youtu.be/first"),
            (1, "https://youtu.be/second"),
        ])
        studio._set_youtube_slot_status = mock.Mock()
        studio._log_msg = mock.Mock()
        studio._prewarm_youtube_queue = mock.Mock()
        studio._youtube_queue_signature = ()
        studio._youtube_queue_prewarmed = set()
        studio._youtube_queue_prewarm_started = False
        studio._youtube_prewarm_wait_started = False

        studio._prepare_youtube_queue("https://youtu.be/first", auto_prewarm=False)
        studio._youtube_queue_prewarmed = {"https://youtu.be/second"}
        studio._youtube_queue_prewarm_started = True
        studio._youtube_prewarm_wait_started = True
        studio._prepare_youtube_queue("https://youtu.be/first", auto_prewarm=False)

        self.assertEqual(studio._youtube_queue_prewarmed, {"https://youtu.be/second"})
        self.assertTrue(studio._youtube_queue_prewarm_started)

    def test_youtube_audio_start_does_not_wait_for_playlist_prewarm(self):
        from avatar_studio import AvatarStudio

        studio = AvatarStudio.__new__(AvatarStudio)
        studio._youtube_primary_url = mock.Mock(return_value="https://youtu.be/first")
        studio._prepare_youtube_queue = mock.Mock()
        studio._youtube_current_slot = mock.Mock(return_value=0)
        studio._set_youtube_slot_status = mock.Mock()
        studio._announce_youtube_link_state = mock.Mock()
        studio.running = False
        studio.engines = None
        studio._attach_youtube_scene = mock.Mock()
        studio._log_msg = mock.Mock()

        studio.speak_youtube_audio()

        studio._prepare_youtube_queue.assert_called_once_with(
            "https://youtu.be/first", auto_prewarm=False)


class JitterBufferTests(unittest.TestCase):
    """The decoder pipe must be drained continuously, not on the display beat.

    Measured over 45 s of a live broadcast, reading one frame at a time at
    display time gave 12.8 fps with eleven freezes of up to 5.1 s, because
    ffmpeg was blocked writing to a full 64 KB pipe and could never fetch
    ahead. Draining into a buffer gave a flat 15.0 fps, longest gap 94 ms.
    """

    FRAME = VIDEO_WIDTH * VIDEO_HEIGHT * 3

    def _stream(self, reads, alive=True, buffer_frames=8):
        proc = mock.Mock()
        proc.poll.return_value = None if alive else 0
        proc.stdout = mock.Mock()
        remaining = list(reads)

        def _read(_stream, size, timeout=None):
            if not remaining:
                proc.poll.return_value = 0
                return b""
            piece = remaining[0]
            head, tail = piece[:size], piece[size:]
            if tail:
                remaining[0] = tail
            else:
                remaining.pop(0)
            return head

        with mock.patch("youtube_video._read_exact", side_effect=_read):
            stream = _DecoderStream(proc, buffer_frames)
            stream._thread.join(timeout=5.0)
        return stream

    def test_reader_thread_assembles_frames_split_across_reads(self):
        half = self.FRAME // 2
        stream = self._stream([
            b"\x40" * half,                  # first half arrives
            b"",                             # reconnect pause, decoder alive
            b"\x40" * (self.FRAME - half),   # the rest arrives
        ])

        self.assertEqual(stream.produced, 1)
        self.assertEqual(len(stream.pop()), self.FRAME)

    def test_reader_thread_keeps_reading_ahead_of_the_display(self):
        # The whole point: three frames are buffered without anyone popping.
        stream = self._stream([b"\x40" * (self.FRAME * 3)])

        self.assertEqual(stream.produced, 3)
        self.assertEqual(stream.depth(), 3)

    def test_full_buffer_drops_the_oldest_frame_and_never_grows(self):
        stream = self._stream([b"\x40" * (self.FRAME * 6)], buffer_frames=4)

        self.assertEqual(stream.depth(), 4)
        self.assertEqual(stream.dropped, 2)

    def test_buffer_reports_eof_when_the_decoder_exits(self):
        stream = self._stream([b"\x40" * self.FRAME], alive=False)

        self.assertTrue(stream.eof)
        self.assertFalse(stream.alive())

    def test_live_source_gets_a_buffer_deep_enough_for_a_segment_stall(self):
        scene = YouTubeVideoScene(lambda: 0.0)
        scene.is_live = True
        scene._source = "https://live.test/playlist.m3u8"

        # Worst observed ffmpeg gap on a live playlist was 3.1 s.
        self.assertGreaterEqual(scene._buffer_frames() / VIDEO_FPS, 3.1)

    def test_local_file_does_not_pay_for_a_deep_buffer(self):
        scene = YouTubeVideoScene(lambda: 0.0)
        scene.is_live = False
        scene._source = r"C:\cache\preview-hd.mp4"

        local = scene._buffer_frames()
        scene._source = "https://example.test/video.mp4"

        self.assertLess(local, scene._buffer_frames())

    def test_buffer_depth_is_configurable(self):
        scene = YouTubeVideoScene(lambda: 0.0)
        scene.is_live = True
        scene._source = "https://live.test/playlist.m3u8"

        with mock.patch.dict(
                os.environ, {"AVATAR_YOUTUBE_VIDEO_BUFFER": "8"}, clear=False):
            self.assertEqual(scene._buffer_frames(), int(round(8 * VIDEO_FPS)))

    def test_buffer_is_never_shallower_than_the_preroll_it_must_hold(self):
        with mock.patch.dict(
                os.environ,
                {"AVATAR_YOUTUBE_LIVE_PREROLL": "6",
                 "AVATAR_YOUTUBE_VIDEO_BUFFER": "1"}, clear=False):
            self.assertGreaterEqual(_video_buffer_seconds(), 6.0)


class LongVideoCachingTests(unittest.TestCase):
    """An ended livestream reports is_live False, so only length can stop it.

    The 5h21m VOD that prompted this was a 1 GB download against a 12 GB cache,
    for a file the scene plays a few minutes of.
    """

    WAS_LIVE_VOD = {
        "title": "Live Futures Trading",
        "duration": 19283,
        "is_live": False,
        "was_live": True,
        "live_status": "was_live",
        "formats": [
            {"url": "https://example.test/video", "vcodec": "h264",
             "height": 720}
        ],
    }

    def _resolve(self, info, env=None):
        class FakeYDL:
            def __init__(self, opts):
                self.opts = opts

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def extract_info(self, _url, download):
                return info

        fake_module = types.SimpleNamespace(YoutubeDL=FakeYDL)
        statuses = []
        with mock.patch.dict(sys.modules, {"yt_dlp": fake_module}), \
                mock.patch.dict(os.environ, env or {}, clear=False), \
                mock.patch("youtube_video._cached_or_download_preview_video",
                           return_value="cache-dir/preview.mp4") as blocking, \
                mock.patch(
                    "youtube_video._start_background_preview_download") as background:
            resolved = resolve_youtube_video(
                "https://youtu.be/test", status_callback=statuses.append)
        return resolved, blocking, background, statuses

    def test_multi_hour_vod_streams_instead_of_caching(self):
        resolved, blocking, background, statuses = self._resolve(self.WAS_LIVE_VOD)

        blocking.assert_not_called()
        background.assert_not_called()
        self.assertEqual(resolved["source"], "https://example.test/video")
        self.assertFalse(resolved["is_live"])
        self.assertTrue(
            any("streaming without caching" in s for s in statuses), statuses)

    def test_short_video_still_uses_the_preview_cache(self):
        info = dict(self.WAS_LIVE_VOD, duration=600)

        resolved, _blocking, background, _statuses = self._resolve(info)

        background.assert_called_once()
        self.assertEqual(resolved["source"], "https://example.test/video")

    def test_cache_limit_is_configurable(self):
        _resolved, _blocking, background, _statuses = self._resolve(
            self.WAS_LIVE_VOD, {"AVATAR_YOUTUBE_CACHE_MAX_MINUTES": "600"})

        background.assert_called_once()

    def test_cache_limit_of_zero_disables_the_length_check(self):
        _resolved, _blocking, background, _statuses = self._resolve(
            self.WAS_LIVE_VOD, {"AVATAR_YOUTUBE_CACHE_MAX_MINUTES": "0"})

        background.assert_called_once()

    def test_default_limit_is_three_hours(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AVATAR_YOUTUBE_CACHE_MAX_MINUTES", None)
            self.assertEqual(_cache_duration_limit(), MAX_CACHE_DURATION_SECONDS)
            self.assertEqual(MAX_CACHE_DURATION_SECONDS, 3 * 3600)

    def test_unknown_duration_is_not_treated_as_too_long(self):
        for info in ({}, {"duration": None}, {"duration": ""},
                     {"duration": "not-a-number"}):
            self.assertFalse(_too_long_to_cache(info), info)

    def test_duration_exactly_at_the_limit_still_caches(self):
        self.assertFalse(_too_long_to_cache({"duration": MAX_CACHE_DURATION_SECONDS}))
        self.assertTrue(_too_long_to_cache({"duration": MAX_CACHE_DURATION_SECONDS + 1}))

    def test_string_duration_is_understood(self):
        self.assertTrue(_too_long_to_cache({"duration": "19283"}))

    def test_queue_preload_gives_up_instead_of_downloading_a_long_vod(self):
        """Preloading wants a local file; it must not fetch a gigabyte to get one."""
        from youtube_video import _cache_preview_video_blocking

        class FakeYDL:
            def __init__(self, opts):
                self.opts = opts

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def extract_info(self, _url, download):
                assert not download, "a long VOD must never be downloaded"
                return self_info

        self_info = self.WAS_LIVE_VOD
        fake_module = types.SimpleNamespace(YoutubeDL=FakeYDL)
        with mock.patch.dict(sys.modules, {"yt_dlp": fake_module}), \
                mock.patch("youtube_video._existing_preview_video", return_value=""), \
                mock.patch(
                    "youtube_video._cached_or_download_preview_video") as blocking:
            result = _cache_preview_video_blocking("https://youtu.be/test")

        blocking.assert_not_called()
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()

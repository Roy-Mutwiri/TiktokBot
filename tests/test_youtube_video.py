import os
import sys
import types
import unittest
from unittest import mock

import numpy as np


ENGINES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "engines")
if ENGINES_DIR not in sys.path:
    sys.path.insert(0, ENGINES_DIR)

from youtube_video import (
    VIDEO_FPS,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
    YouTubeVideoScene,
    _decoder_needs_restart,
    _download_tuning_opts,
    _ffmpeg_header_text,
    _is_usable_preview_file,
    _preview_video_candidates,
    _select_video_source,
    normalized_crop,
    resolve_youtube_video,
)


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

    def _live_scene_with_reads(self, reads, alive=True):
        """Run the scene loop against a scripted decoder read sequence."""
        statuses = []
        scene = YouTubeVideoScene(lambda: 0.0, status_callback=statuses.append)
        scene.statuses = statuses
        scene._running = True
        proc = mock.Mock()
        proc.poll.return_value = None if alive else 1
        proc.stdout = mock.Mock()
        opened = []

        def _open(_position):
            opened.append(1)
            return proc

        remaining = list(reads)

        def _read(_stream, size, timeout=None):
            if not remaining:
                scene._running = False
                return b""
            piece = remaining.pop(0)
            if not remaining:
                # Let the loop finish this frame, then stop it.
                scene._running_after_frame = True
            return piece[:size]

        resolved = {
            "source": "https://live.test/playlist.m3u8",
            "direct_source": "https://live.test/playlist.m3u8",
            "title": "Live", "duration": 0.0, "headers": {},
            "direct_headers": {}, "is_live": True, "thumbnail": "",
        }
        with mock.patch("youtube_video.resolve_youtube_video",
                        return_value=resolved), \
                mock.patch.object(scene, "_open_decoder", side_effect=_open), \
                mock.patch("youtube_video._read_exact", side_effect=_read), \
                mock.patch("youtube_video._thumbnail_frame", return_value=None):
            scene._run()
        return scene, proc, len(opened)

    def test_live_decoder_survives_a_reconnect_pause_mid_frame(self):
        # ffmpeg goes quiet for a few seconds while it reconnects a dropped TLS
        # connection. The half-read frame must be kept, not thrown away with a
        # decoder that is still alive.
        frame_bytes = VIDEO_WIDTH * VIDEO_HEIGHT * 3
        half = frame_bytes // 2
        scene, proc, opened = self._live_scene_with_reads([
            b"\x40" * half,          # first half arrives
            b"",                     # reconnect pause - nothing to read
            b"\x40" * (frame_bytes - half),   # the rest arrives
        ])

        self.assertEqual(opened, 1)          # decoder was never restarted
        self.assertTrue(scene.video_ready)
        self.assertEqual(scene.latest_frame.shape,
                         (VIDEO_HEIGHT, VIDEO_WIDTH, 3))

    def test_replacing_a_scene_is_not_reported_as_a_stream_failure(self):
        frame_bytes = VIDEO_WIDTH * VIDEO_HEIGHT * 3
        statuses = []
        scene = YouTubeVideoScene(lambda: 0.0, status_callback=statuses.append)
        scene._running = True
        proc = mock.Mock()
        proc.poll.return_value = 1
        proc.stdout = mock.Mock()

        def _read(_stream, _size, timeout=None):
            # stop() lands while the decoder is mid-frame, exactly as it does
            # when a new link replaces the scene.
            scene._running = False
            return b"\x40" * (frame_bytes // 2)

        resolved = {
            "source": "https://live.test/playlist.m3u8",
            "direct_source": "https://live.test/playlist.m3u8",
            "title": "Live", "duration": 0.0, "headers": {},
            "direct_headers": {}, "is_live": True, "thumbnail": "",
        }
        with mock.patch("youtube_video.resolve_youtube_video",
                        return_value=resolved), \
                mock.patch.object(scene, "_open_decoder", return_value=proc), \
                mock.patch("youtube_video._read_exact", side_effect=_read), \
                mock.patch("youtube_video._thumbnail_frame", return_value=None):
            scene._run()

        self.assertFalse([s for s in statuses if "hiccup" in s or "resyncing" in s],
                         statuses)

    def test_live_decoder_restarts_when_the_process_actually_died(self):
        frame_bytes = VIDEO_WIDTH * VIDEO_HEIGHT * 3
        scene, _proc, opened = self._live_scene_with_reads(
            [b"\x40" * (frame_bytes // 2), b""], alive=False)

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


if __name__ == "__main__":
    unittest.main()

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
    YouTubeVideoScene,
    _decoder_needs_restart,
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
                return "https://youtu.be/Wpj5TYGw4cY\n"

        studio = AvatarStudio.__new__(AvatarStudio)
        studio.youtube_entry = FakeEntry()
        studio._attach_youtube_scene = mock.Mock()
        studio._start_scene_snip = mock.Mock()

        studio._add_scene()

        studio._attach_youtube_scene.assert_called_once_with(
            "https://youtu.be/Wpj5TYGw4cY", force=True)
        studio._start_scene_snip.assert_not_called()


if __name__ == "__main__":
    unittest.main()

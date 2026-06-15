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
    _select_video_source,
    normalized_crop,
    resolve_youtube_video,
)


class YouTubeVideoTests(unittest.TestCase):
    def test_video_scene_uses_smooth_preview_frame_rate(self):
        self.assertGreaterEqual(VIDEO_FPS, 24.0)

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
            ]
        }
        self.assertEqual(_select_video_source(info), "720")

    def test_video_source_prefers_progressive_h264_for_preview(self):
        info = {
            "formats": [
                {"url": "dash-720", "vcodec": "avc1.64001f",
                 "acodec": "none", "height": 720, "fps": 30},
                {"url": "progressive-360", "vcodec": "avc1.42001e",
                 "acodec": "mp4a.40.2", "height": 360, "fps": 30},
            ]
        }

        self.assertEqual(_select_video_source(info), "progressive-360")

    def test_normalized_crop_uses_fractional_coordinates(self):
        frame = np.arange(100 * 200 * 3, dtype=np.uint8).reshape(100, 200, 3)
        cropped = normalized_crop(frame, (0.25, 0.20, 0.75, 0.80))
        self.assertEqual(cropped.shape, (60, 100, 3))
        np.testing.assert_array_equal(cropped[0, 0], frame[20, 50])

    def test_resolver_returns_video_metadata_without_download(self):
        info = {
            "title": "Test video",
            "duration": 123,
            "http_headers": {"User-Agent": "test"},
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
                self.download = download
                return info

        fake_module = types.SimpleNamespace(YoutubeDL=FakeYDL)
        with mock.patch.dict(sys.modules, {"yt_dlp": fake_module}):
            resolved = resolve_youtube_video("https://youtu.be/test")

        self.assertEqual(resolved["source"], "https://example.test/video")
        self.assertEqual(resolved["title"], "Test video")
        self.assertEqual(resolved["duration"], 123.0)

    def test_studio_video_and_audio_use_the_same_position_clock(self):
        from avatar_studio import AvatarStudio

        studio = AvatarStudio.__new__(AvatarStudio)
        studio._youtube_audio = types.SimpleNamespace(position_seconds=42.75)
        studio._youtube_start_seconds = 0.0
        studio._youtube_chunks = []

        self.assertEqual(studio._youtube_position_seconds(), 42.75)

    def test_caption_clock_uses_video_duration_when_audio_is_absent(self):
        from avatar_studio import AvatarStudio

        studio = AvatarStudio.__new__(AvatarStudio)
        studio._youtube_audio = None
        studio._youtube_start_seconds = 10.0
        studio._youtube_end_seconds = None
        studio._youtube_duration = 0.0
        studio._youtube_chunks = ["a", "b", "c", "d"]
        studio._youtube_index = 2
        studio._youtube_scene = types.SimpleNamespace(duration=110.0)

        self.assertEqual(studio._youtube_position_seconds(), 60.0)

    def test_decoder_uses_hybrid_seek_and_safe_user_agent(self):
        scene = YouTubeVideoScene(lambda: 45.0)
        scene._source = "https://example.test/video.mp4"
        scene._headers = {
            "User-Agent": "browser",
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
        self.assertNotIn("-headers", cmd)

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

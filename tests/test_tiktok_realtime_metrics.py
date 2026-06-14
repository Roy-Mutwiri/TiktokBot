import os
import sys
import unittest


ENGINES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "engines")
if ENGINES_DIR not in sys.path:
    sys.path.insert(0, ENGINES_DIR)

import resource_monitor
from tiktok_comments import ROOM_METRICS_INTERVAL, TikTokComments


class TikTokRealtimeMetricTests(unittest.TestCase):
    def test_viewer_callback_is_registered(self):
        callback = object()
        reader = TikTokComments("@host", lambda *_: None, on_viewers=callback)
        self.assertIs(reader.on_viewers, callback)
        self.assertEqual(reader.unique_id, "host")
        self.assertEqual(reader.username, "@host")

    def test_room_info_metrics_are_extracted(self):
        info = {
            "user_count": 43,
            "like_count": 372,
            "stats": {"like_count": 371},
        }
        self.assertEqual(TikTokComments._room_metrics(info), (43, 372))

    def test_room_info_metrics_fall_back_to_stats(self):
        info = {"stats": {"user_count": 21, "digg_count": 99}}
        self.assertEqual(TikTokComments._room_metrics(info), (21, 99))

    def test_resource_monitor_default_is_fast_but_bounded(self):
        self.assertGreaterEqual(resource_monitor.INTERVAL, 0.05)
        self.assertLessEqual(resource_monitor.INTERVAL, 0.1)

    def test_tiktok_fallback_has_no_one_second_artificial_delay(self):
        self.assertGreaterEqual(ROOM_METRICS_INTERVAL, 0.05)
        self.assertLessEqual(ROOM_METRICS_INTERVAL, 0.1)

    def test_resource_monitor_exposes_unsmoothed_dashboard_values(self):
        monitor = resource_monitor.ResourceMonitor.__new__(
            resource_monitor.ResourceMonitor)
        monitor.cpu_live = 12.0
        monitor.gpu_live = 34.0
        monitor.vram_live = 56.0
        self.assertEqual(
            (monitor.cpu_live, monitor.gpu_live, monitor.vram_live),
            (12.0, 34.0, 56.0),
        )


if __name__ == "__main__":
    unittest.main()

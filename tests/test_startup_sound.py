import os
import unittest
from unittest import mock

import startup_sound


class StartupSoundTests(unittest.TestCase):
    def test_live_status_sounds_are_distinct(self):
        online, online_rate = startup_sound._synth_live_online()
        offline, offline_rate = startup_sound._synth_live_offline()

        self.assertEqual(online_rate, startup_sound.SR)
        self.assertEqual(offline_rate, startup_sound.SR)
        self.assertNotEqual(online.shape, offline.shape)
        self.assertGreater(float(abs(online).max()), 0.1)
        self.assertGreater(float(abs(offline).max()), 0.1)

    def test_live_status_sound_toggle(self):
        with mock.patch.dict(os.environ, {"AVATAR_LIVE_STATUS_SOUNDS": "0"}), \
                mock.patch("startup_sound.threading.Thread") as thread:
            startup_sound.play_live_online_sound()
            startup_sound.play_live_offline_sound()
        thread.assert_not_called()


if __name__ == "__main__":
    unittest.main()

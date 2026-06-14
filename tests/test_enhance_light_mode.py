import unittest
from unittest import mock
import os
import sys

import numpy as np

ENGINES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "engines")
if ENGINES_DIR not in sys.path:
    sys.path.insert(0, ENGINES_DIR)

import enhance_engine


class EnhanceLightModeTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((512, 512, 3), 96, dtype=np.uint8)

    def tearDown(self):
        enhance_engine.set_level("full")

    def test_light_mode_skips_background_segmentation(self):
        enhance_engine.set_level("light")
        with mock.patch.object(
            enhance_engine, "background_composite", side_effect=AssertionError
        ):
            out = enhance_engine.enhance_frame(self.frame)

        self.assertEqual(out.shape, self.frame.shape)

    def test_full_mode_keeps_background_segmentation(self):
        enhance_engine.set_level("full")
        with mock.patch.object(
            enhance_engine, "background_composite", return_value=self.frame
        ) as bg:
            enhance_engine.enhance_frame(self.frame)

        bg.assert_called_once()


if __name__ == "__main__":
    unittest.main()

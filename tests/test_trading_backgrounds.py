import unittest

import numpy as np

from engines.trading_backgrounds import (
    BACKGROUND_PRESETS,
    apply_subject_lighting,
    render_background,
)


class TradingBackgroundTests(unittest.TestCase):
    def test_has_more_than_one_hundred_background_options(self):
        self.assertGreaterEqual(len(BACKGROUND_PRESETS), 101)
        self.assertEqual(BACKGROUND_PRESETS[0], "No Background")

    def test_no_background_returns_none(self):
        self.assertIsNone(render_background("No Background", (128, 128)))

    def test_presets_render_at_requested_size(self):
        for name in BACKGROUND_PRESETS[1::20]:
            frame = render_background(name, (160, 96), phase=0.5)
            self.assertEqual(frame.shape, (96, 160, 3))
            self.assertEqual(frame.dtype, np.uint8)
            self.assertGreater(float(frame.std()), 4.0)

    def test_lighting_never_changes_pixels_outside_subject_mask(self):
        frame = np.full((80, 100, 3), 100, dtype=np.uint8)
        mask = np.zeros((80, 100, 1), dtype=np.float32)
        mask[20:70, 30:75] = 1.0

        lit = apply_subject_lighting(
            frame, mask, "Crypto Command Center / Teal Magenta", strength=0.4)

        np.testing.assert_array_equal(lit[:15], frame[:15])
        np.testing.assert_array_equal(lit[:, :25], frame[:, :25])
        self.assertFalse(np.array_equal(lit[30:60, 40:65], frame[30:60, 40:65]))


if __name__ == "__main__":
    unittest.main()

import os
import sys
import unittest

import numpy as np


ENGINES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "engines")
if ENGINES_DIR not in sys.path:
    sys.path.insert(0, ENGINES_DIR)

from voice_changer_engine import BLOCK, make_converter


class VoiceChangerTests(unittest.TestCase):
    def test_arabic_light_converter_keeps_block_shape_and_bounds(self):
        conv = make_converter("arabic-light")
        x = np.linspace(-0.6, 0.6, BLOCK, dtype=np.float32)

        y = conv.convert(x)

        self.assertEqual(y.dtype, np.float32)
        self.assertEqual(y.shape, x.shape)
        self.assertLessEqual(float(np.max(np.abs(y))), 1.0)
        self.assertGreater(float(np.max(np.abs(y))), 0.1)

    def test_youtube_disguise_preserves_audio_block_contract(self):
        conv = make_converter("youtube-disguise")
        phase = np.linspace(0.0, np.pi * 8.0, BLOCK, dtype=np.float32)
        x = (0.3 * np.sin(phase)).astype(np.float32)

        outputs = [conv.convert(x) for _ in range(2)]
        audible = [
            y for y in outputs
            if y is not None and len(y) > 0
            and float(np.max(np.abs(y))) > 0.001
        ]

        self.assertTrue(audible)
        self.assertEqual(audible[-1].dtype, np.float32)
        self.assertEqual(audible[-1].shape, x.shape)
        self.assertLessEqual(float(np.max(np.abs(audible[-1]))), 1.0)


if __name__ == "__main__":
    unittest.main()

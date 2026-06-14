import unittest

import numpy as np

from engines import wav2lip_engine


class Wav2LipRawOutputTests(unittest.TestCase):
    def test_raw_output_skips_internal_blend(self):
        engine = wav2lip_engine.Wav2LipEngine.__new__(
            wav2lip_engine.Wav2LipEngine
        )
        engine.fallback = False
        engine.ready = True
        engine._err_printed = False
        engine._audio_active = lambda: True
        engine._get_face_box = lambda frame: (10, 10, 50, 50)
        engine._build_mel = lambda: object()
        engine._run_wav2lip = lambda crop, mel: np.full_like(crop, 220)

        frame = np.full((64, 64, 3), 40, dtype=np.uint8)
        raw = engine.process_frame(frame, raw_output=True)
        blended = engine.process_frame(frame, raw_output=False)

        np.testing.assert_array_equal(raw[10:50, 10:50], 220)
        self.assertLess(float(np.mean(blended[10:50, 10:50])), 220.0)
        np.testing.assert_array_equal(raw[:8], frame[:8])


if __name__ == "__main__":
    unittest.main()

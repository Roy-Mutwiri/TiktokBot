import unittest

import cv2
import numpy as np

from engines import compositor


class MouthCompositorTests(unittest.TestCase):
    def test_mouth_hint_box_is_tight_around_lips(self):
        comp = compositor.Compositor()
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        bbox = comp.detect_mouth_bbox(frame, mouth_hint=(150, 170, 60))
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        self.assertLess(height, width * 0.65)
        self.assertLess(height, 60)
        self.assertGreater(width, 75)

    def test_exclusive_mode_replaces_native_mouth_center(self):
        comp = compositor.Compositor()
        frame = np.full((160, 160, 3), 150, dtype=np.uint8)
        bbox = (40, 55, 120, 125)

        # Simulate a visible native mouth that must not survive under AI lips.
        cv2.rectangle(frame, (55, 82), (105, 100), (10, 10, 10), -1)
        ai_mouth = np.full((70, 80, 3), (70, 90, 180), dtype=np.uint8)

        out = comp.blend_mouth(frame, ai_mouth, bbox, exclusive=True)

        center = out[90, 80].astype(np.int16)
        native = frame[90, 80].astype(np.int16)
        self.assertGreater(np.linalg.norm(center - native), 80)
        self.assertGreater(int(center[2]), int(center[0]))

    def test_exclusive_mode_keeps_outer_frame_unchanged(self):
        comp = compositor.Compositor()
        frame = np.full((100, 100, 3), 120, dtype=np.uint8)
        mouth = np.full((40, 50, 3), 220, dtype=np.uint8)

        out = comp.blend_mouth(frame, mouth, (25, 35, 75, 75), exclusive=True)

        np.testing.assert_array_equal(out[:30], frame[:30])
        np.testing.assert_array_equal(out[80:], frame[80:])

    def test_detail_transfer_preserves_edge_texture_not_native_lips(self):
        comp = compositor.Compositor()
        frame = np.full((120, 120, 3), 130, dtype=np.uint8)
        bbox = (30, 40, 90, 90)
        region = frame[40:90, 30:90]
        region[:, ::2] = 175
        cv2.rectangle(frame, (48, 60), (72, 72), (5, 5, 5), -1)
        generated = np.full((50, 60, 3), (80, 100, 180), dtype=np.uint8)

        out = comp.blend_mouth(frame, generated, bbox, exclusive=True)

        edge_variation = float(np.std(out[48:80, 33:43]))
        center = out[66, 60].astype(np.int16)
        native = frame[66, 60].astype(np.int16)
        self.assertGreater(edge_variation, 5.0)
        self.assertGreater(np.linalg.norm(center - native), 80)

if __name__ == "__main__":
    unittest.main()

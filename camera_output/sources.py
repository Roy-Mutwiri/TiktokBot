"""
sources.py — frame producers for the camera output pipeline. Each source
yields BGR numpy frames at the requested resolution; the controller converts
and writes them to the ring.

  TestPatternSource : SMPTE-ish color bars + live timestamp + moving counter
                      (proves end-to-end motion without any external input)
  StaticImageSource : loops a single image file
  VideoFileSource   : loops a video file at its own pace
  CallableSource    : wraps any callable returning a BGR frame (app hook)
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


class FrameSource:
    name = "base"

    def read(self) -> Optional[np.ndarray]:
        raise NotImplementedError

    def close(self):
        pass


class TestPatternSource(FrameSource):
    name = "test-pattern"

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self.w, self.h, self.fps = width, height, fps
        self._n = 0
        self._bars = self._make_bars(width, height)

    @staticmethod
    def _make_bars(w: int, h: int) -> np.ndarray:
        # 8 vertical color bars (BGR).
        colors = [
            (255, 255, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0),
            (255, 0, 255), (0, 0, 255), (255, 0, 0), (0, 0, 0),
        ]
        img = np.zeros((h, w, 3), dtype=np.uint8)
        bw = w // len(colors)
        for i, c in enumerate(colors):
            x0 = i * bw
            x1 = w if i == len(colors) - 1 else (i + 1) * bw
            img[:, x0:x1] = c
        return img

    def read(self) -> np.ndarray:
        img = self._bars.copy()
        self._n += 1
        # Moving bar to prove motion / detect freezes.
        x = int((self._n * 6) % self.w)
        cv2.rectangle(img, (x, 0), (min(x + 8, self.w), self.h), (40, 40, 40), -1)
        if cv2 is not None:
            ts = time.strftime("%H:%M:%S")
            txt = f"RoyCam TEST  {ts}  #{self._n}"
            cv2.rectangle(img, (0, self.h - 60), (self.w, self.h), (0, 0, 0), -1)
            cv2.putText(img, txt, (20, self.h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
                        cv2.LINE_AA)
        return img


class StaticImageSource(FrameSource):
    name = "static-image"

    def __init__(self, path: str, width: int = 1280, height: int = 720):
        if cv2 is None:
            raise RuntimeError("cv2 required for StaticImageSource")
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)
        self._img = cv2.resize(img, (width, height))

    def read(self) -> np.ndarray:
        return self._img.copy()


class VideoFileSource(FrameSource):
    name = "video-file"

    def __init__(self, path: str, width: int = 1280, height: int = 720,
                 loop: bool = True):
        if cv2 is None:
            raise RuntimeError("cv2 required for VideoFileSource")
        self.path, self.w, self.h, self.loop = path, width, height, loop
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise FileNotFoundError(path)

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self._cap.read()
        if not ok:
            if self.loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
            if not ok:
                return None
        return cv2.resize(frame, (self.w, self.h))

    def close(self):
        self._cap.release()


class CallableSource(FrameSource):
    """Adapter so the live app can push its own rendered BGR frames."""
    name = "callable"

    def __init__(self, fn: Callable[[], Optional[np.ndarray]]):
        self._fn = fn

    def read(self) -> Optional[np.ndarray]:
        return self._fn()

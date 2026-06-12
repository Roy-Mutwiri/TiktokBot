"""
controller.py — drives a FrameSource at a fixed fps, converts each frame to the
advertised pixel format, and writes it into the shared-memory ring.

This is the app-side half of the Phase-1 pipeline. The service reads the same
ring and (eventually) hands frames to the AVStream driver; for now the service
just validates + previews them.

Designed to be embedded in the Tkinter app (run on a daemon thread) or run
standalone from camera_output.__main__.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

from . import frame_converter as conv
from . import roycam_format as F
from .ring import RingWriter
from .sources import FrameSource, TestPatternSource


class CameraOutputController:
    def __init__(self, source: Optional[FrameSource] = None,
                 width: int = 1280, height: int = 720,
                 fmt_id: int = F.FMT_NV12, fps: int = 30,
                 max_width: int = 1920, max_height: int = 1080,
                 path: Optional[str] = None):
        self.width = width
        self.height = height
        self.fmt_id = fmt_id
        self.fps = fps
        self.source = source or TestPatternSource(width, height, fps)
        self.writer = RingWriter(max_width, max_height, fmt_id, path=path)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.frames_written = 0
        self.last_error: Optional[str] = None

    # --- lifecycle -----------------------------------------------------------
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="roycam-output")
        self._thread.start()

    def stop(self, timeout: float = 2.0):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)
        self.writer.close()
        try:
            self.source.close()
        except Exception:
            pass

    def set_source(self, source: FrameSource):
        old = self.source
        self.source = source
        if old is not None and old is not source:
            try:
                old.close()
            except Exception:
                pass

    # --- hot loop ------------------------------------------------------------
    def _fallback_frame(self) -> np.ndarray:
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:] = (20, 20, 20)
        return img

    def _run(self):
        period = 1.0 / max(1, self.fps)
        next_t = time.perf_counter()
        while not self._stop.is_set():
            try:
                frame = self.source.read()
                fallback = frame is None
                if fallback:
                    frame = self._fallback_frame()
                payload, w, h, _ = conv.convert(frame, self.fmt_id)
                self.writer.write(payload, self.fmt_id, w, h,
                                  fps_num=self.fps, fps_den=1,
                                  fallback=fallback)
                self.frames_written += 1
                self.last_error = None
            except Exception as e:    # never let the camera loop die
                self.last_error = repr(e)
            # Fixed-rate pacing (latest-frame-wins; no catch-up bursts).
            next_t += period
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                self._stop.wait(sleep)
            else:
                next_t = time.perf_counter()

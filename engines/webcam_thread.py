# =============================================================================
# engines/webcam_thread.py
# -----------------------------------------------------------------------------
# Threaded webcam capture that always holds the FRESHEST frame.
#
# OpenCV buffers frames internally; if the render loop is slower than the camera
# (e.g. 17fps loop vs 30fps cam), cap.read() returns OLD buffered frames, so the
# avatar reacts to where your head WAS, not where it IS — visible lag, especially
# on turns. This grabber runs in a background thread, continuously draining the
# camera and keeping only the latest frame, so the loop reads the current frame
# instantly (no blocking, no buffer latency). Wraps an already-opened VideoCapture.
# =============================================================================

import threading
import time


class ThreadedCamera:
    """Background grabber: keeps only the most recent frame (low latency)."""

    def __init__(self, cap):
        self._cap = cap
        self._frame = None
        self._ok = False
        self._lock = threading.Lock()
        self._running = True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        # wait briefly for the first frame
        for _ in range(50):
            with self._lock:
                if self._frame is not None:
                    break
            time.sleep(0.01)

    def _loop(self):
        while self._running:
            ok, frame = self._cap.read()
            if ok and frame is not None:
                with self._lock:
                    self._frame = frame
                    self._ok = True
            else:
                time.sleep(0.005)

    def read(self):
        """Return (ok, freshest_frame) without blocking on the camera."""
        with self._lock:
            if self._frame is None:
                return False, None
            return self._ok, self._frame.copy()

    def isOpened(self):
        return self._cap is not None and self._cap.isOpened()

    def release(self):
        self._running = False
        try:
            self._t.join(timeout=1.0)
        except Exception:
            pass
        try:
            self._cap.release()
        except Exception:
            pass

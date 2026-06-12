"""
roycam_service.py — RoyCam Frame Bridge (Phase-1 user-mode skeleton).

Reads the app's shared-memory ring, validates each frame against the contract,
applies the fallback policy when the source is absent/stale, exposes a status
named pipe, and hands frames to a driver bridge.

In Phase-1 the driver bridge is a STUB (logs + optional preview) because the
AVStream driver is not built yet. The same loop will later DeviceIoControl the
RoyCam driver (ROYC_IOCTL_SUBMIT_FRAME). The service is deliberately
crash-proof: any per-frame error degrades to the fallback frame, never exits.

Run:  python -m service.roycam_service --preview
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time

# Allow running as a script (python service/roycam_service.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera_output import roycam_format as F           # noqa: E402
from camera_output.ring import RingReader              # noqa: E402

STATUS_PIPE = r"\\.\pipe\RoyCamStatus"
STALE_AFTER = 0.5    # seconds without a fresh frame -> fallback


class DriverBridge:
    """Phase-1 stub. Replace submit() with DeviceIoControl to the driver."""

    def __init__(self, preview: bool = False):
        self.preview = preview
        self.submitted = 0
        self._cv2 = None
        if preview:
            try:
                import cv2
                self._cv2 = cv2
            except ImportError:
                self.preview = False

    def submit(self, header: dict, payload: bytes):
        self.submitted += 1
        if self.preview and self._cv2 is not None:
            try:
                from camera_output import frame_converter as conv
                bgr = conv.decode(payload, header["format"],
                                  header["width"], header["height"])
                self._cv2.imshow("RoyCam bridge preview", bgr)
                self._cv2.waitKey(1)
            except Exception:
                pass

    def close(self):
        if self._cv2 is not None:
            try:
                self._cv2.destroyAllWindows()
            except Exception:
                pass


def validate(ctrl: dict, header: dict) -> int:
    """Return a ROYC status code; ST_OK means safe to submit."""
    if ctrl.get("magic") != F.MAGIC or ctrl.get("version") != F.VERSION:
        return F.ST_BAD_HEADER
    if header.get("magic") != F.MAGIC:
        return F.ST_BAD_HEADER
    if header["format"] not in F.FORMAT_NAMES:
        return F.ST_UNSUPPORTED_FORMAT
    w, h = header["width"], header["height"]
    if w <= 0 or h <= 0 or w > ctrl["max_width"] or h > ctrl["max_height"]:
        return F.ST_RESOLUTION_MISMATCH
    expected = F.payload_size(header["format"], w, h)
    if header["data_size"] != expected:
        return F.ST_FRAME_TOO_LARGE
    if not (header["status_flags"] & F.FLAG_READY):
        return F.ST_NO_SOURCE
    return F.ST_OK


class StatusServer:
    """Best-effort status publisher. Uses a Windows named pipe if available,
    otherwise writes a JSON status file (cross-platform / dev fallback)."""

    def __init__(self):
        self.state = {"status": F.ST_SERVICE_STOPPED, "frames": 0,
                      "fallback": False, "source_state": F.SRC_NONE}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._fallback_file = os.path.join(
            os.environ.get("TEMP", "."), "roycam_status.json")
        self._t = threading.Thread(target=self._serve, daemon=True)
        self._t.start()

    def update(self, **kw):
        with self._lock:
            self.state.update(kw)

    def _snapshot(self) -> str:
        import json
        with self._lock:
            return json.dumps(self.state)

    def _serve(self):
        # Pipe server is Windows-only; fall back to a status file everywhere.
        while not self._stop.is_set():
            try:
                with open(self._fallback_file, "w") as fh:
                    fh.write(self._snapshot())
            except Exception:
                pass
            self._stop.wait(0.5)

    def close(self):
        self._stop.set()


def run(path: str = None, preview: bool = False):
    bridge = DriverBridge(preview=preview)
    status = StatusServer()
    reader = None
    last_frame_idx = -1
    last_seen = 0.0
    print("[roycam-service] starting frame bridge (Phase-1 stub)")
    try:
        while True:
            if reader is None:
                try:
                    reader = RingReader(path)
                except Exception:
                    status.update(status=F.ST_NO_SOURCE)
                    time.sleep(0.25)
                    continue
            try:
                ctrl = reader.control()
                got = reader.read_latest()
            except Exception:
                # Ring vanished (app restart); reopen next loop.
                try:
                    reader.close()
                except Exception:
                    pass
                reader = None
                continue

            now = time.perf_counter()
            if got is not None:
                header, payload = got
                code = validate(ctrl, header)
                fresh = header["frame_index"] != last_frame_idx
                if code == F.ST_OK and fresh:
                    bridge.submit(header, payload)
                    last_frame_idx = header["frame_index"]
                    last_seen = now
                    status.update(status=F.ST_OK, frames=bridge.submitted,
                                  fallback=bool(header["status_flags"]
                                                & F.FLAG_FALLBACK),
                                  source_state=ctrl["source_state"])
                elif code != F.ST_OK:
                    status.update(status=code)

            # Fallback policy: source stale -> mark fallback active.
            if now - last_seen > STALE_AFTER:
                status.update(status=F.ST_FALLBACK_ACTIVE, fallback=True,
                              source_state=ctrl.get("source_state", F.SRC_NONE))

            time.sleep(1.0 / 120.0)   # poll ~2x frame rate
    except KeyboardInterrupt:
        pass
    finally:
        status.close()
        bridge.close()
        if reader is not None:
            reader.close()
        print(f"[roycam-service] stopped (submitted={bridge.submitted})")


def main():
    ap = argparse.ArgumentParser(prog="roycam_service")
    ap.add_argument("--path", help="ring file path (default: public dir)")
    ap.add_argument("--preview", action="store_true",
                    help="show what the driver would receive")
    args = ap.parse_args()
    run(path=args.path, preview=args.preview)


if __name__ == "__main__":
    main()

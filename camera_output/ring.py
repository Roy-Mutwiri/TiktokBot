"""
ring.py — triple-buffer shared-memory ring (app writer / service reader).

Layout matches shared/roycam_frame_format.h:
    [ control 64B ][ slot0 ][ slot1 ][ slot2 ]
    slot = [ header 128B ][ payload (slot_stride - 128) ]

Tear-free protocol:
  * Writer rotates to slot = (write_index + 1) % 3 — never the slot a reader
    is most likely reading (the published write_index).
  * Per-slot SEQLOCK: seq is bumped odd before the body write and even after,
    so a reader that sees a stable even seq before and after knows the slot
    didn't change mid-read.
  * write_index is published last, so readers always land on a completed slot.

The MMF is backed by a file under a public dir so a future Windows service
(running in its own session) can open the same mapping by name/path.
"""
from __future__ import annotations

import ctypes
import mmap
import os
import time
from typing import Optional, Tuple

from . import roycam_format as F

DEFAULT_DIR  = r"C:\Users\Public\RoyCamCamera"
DEFAULT_FILE = "roycam_ring.bin"


def _qpc() -> int:
    """QueryPerformanceCounter on Windows, perf_counter_ns elsewhere."""
    try:
        c = ctypes.c_int64(0)
        if ctypes.windll.kernel32.QueryPerformanceCounter(ctypes.byref(c)):
            return c.value
    except Exception:
        pass
    return time.perf_counter_ns()


class RingWriter:
    """App-side producer. Owns the control block and writes frame slots."""

    def __init__(self, max_w: int, max_h: int, fmt_id: int = F.FMT_NV12,
                 path: Optional[str] = None):
        self.max_w = max_w
        self.max_h = max_h
        self.fmt_id = fmt_id
        # slot must fit the largest advertised format at max resolution.
        max_payload = max(F.payload_size(f, max_w, max_h)
                          for f in (F.FMT_NV12, F.FMT_YUY2, F.FMT_RGB32))
        self.slot_stride = F.HEADER_SZ + max_payload
        self.map_size = F.total_map_size(self.slot_stride)

        d = os.path.dirname(path) if path else DEFAULT_DIR
        self.path = path or os.path.join(DEFAULT_DIR, DEFAULT_FILE)
        os.makedirs(d, exist_ok=True)
        # Pre-size the backing file.
        with open(self.path, "wb") as fh:
            fh.truncate(self.map_size)
        self._fh = open(self.path, "r+b")
        self._mm = mmap.mmap(self._fh.fileno(), self.map_size)

        self._frame_index = 0
        self._write_index = F.RING_SLOTS - 1   # first frame -> slot 0
        self._seq = [0] * F.RING_SLOTS
        self._publish_control(F.SRC_NONE)

    def _publish_control(self, source_state: int):
        self._mm[0:F.CONTROL_SZ] = F.pack_control(
            self.slot_stride, self.max_w, self.max_h,
            self._write_index, source_state)

    def set_source_state(self, state: int):
        self._publish_control(state)

    def write(self, payload: bytes, fmt_id: int, w: int, h: int,
              fps_num: int = 30, fps_den: int = 1,
              fallback: bool = False) -> int:
        """Write one frame to the next slot; returns the slot index used."""
        if len(payload) > self.slot_stride - F.HEADER_SZ:
            raise ValueError("payload exceeds slot capacity")
        slot = (self._write_index + 1) % F.RING_SLOTS
        base = F.slot_offset(self.slot_stride, slot)

        # SEQLOCK: bump to odd (writing in progress).
        self._seq[slot] += 1
        odd = self._seq[slot]
        flags = F.FLAG_READY | (F.FLAG_FALLBACK if fallback else 0)
        self._mm[base:base + F.HEADER_SZ] = F.pack_header(
            fmt_id, w, h, F.stride_for(fmt_id, w), len(payload),
            fps_num, fps_den, self._frame_index, _qpc(), odd, flags)

        # Body write.
        pstart = base + F.HEADER_SZ
        self._mm[pstart:pstart + len(payload)] = payload

        # SEQLOCK: bump to even (stable) — rewrite header seq field only.
        self._seq[slot] += 1
        even = self._seq[slot]
        self._mm[base:base + F.HEADER_SZ] = F.pack_header(
            fmt_id, w, h, F.stride_for(fmt_id, w), len(payload),
            fps_num, fps_den, self._frame_index, _qpc(), even, flags)

        # Publish.
        self._write_index = slot
        self._frame_index += 1
        self._publish_control(F.SRC_ACTIVE if not fallback else F.SRC_PAUSED)
        return slot

    def close(self):
        try:
            self.set_source_state(F.SRC_NONE)
        except Exception:
            pass
        try:
            self._mm.close()
        finally:
            self._fh.close()


class RingReader:
    """Service/test-side consumer. Reads the latest completed slot tear-free."""

    def __init__(self, path: Optional[str] = None):
        self.path = path or os.path.join(DEFAULT_DIR, DEFAULT_FILE)
        self._fh = open(self.path, "r+b")
        self._mm = mmap.mmap(self._fh.fileno(), 0)

    def control(self) -> dict:
        return F.unpack_control(self._mm[0:F.CONTROL_SZ])

    def read_latest(self, retries: int = 8) -> Optional[Tuple[dict, bytes]]:
        """Return (header, payload) of the latest stable frame, or None."""
        ctrl = self.control()
        if ctrl["magic"] != F.MAGIC:
            return None
        stride = ctrl["slot_stride"]
        slot = ctrl["write_index"]
        if slot >= F.RING_SLOTS:
            return None
        base = F.slot_offset(stride, slot)

        for _ in range(retries):
            h1 = F.unpack_header(self._mm[base:base + F.HEADER_SZ])
            if h1["magic"] != F.MAGIC or not (h1["status_flags"] & F.FLAG_READY):
                return None
            if h1["seq"] & 1:        # writer mid-write
                continue
            size = h1["data_size"]
            pstart = base + F.HEADER_SZ
            payload = bytes(self._mm[pstart:pstart + size])
            h2 = F.unpack_header(self._mm[base:base + F.HEADER_SZ])
            if h2["seq"] == h1["seq"]:   # stable across the body read
                return h1, payload
        return None

    def close(self):
        try:
            self._mm.close()
        finally:
            self._fh.close()

"""
Phase-1 pipeline tests. Runnable with pytest OR directly:

    python tests/test_camera_output.py

Covers:
  * frame contract sizes match the C header
  * BGR -> {NV12,YUY2,RGB32} -> BGR round-trip within a mean-abs-error bound
  * triple-buffer ring: writer/reader agree, no torn frames under a 30fps load
"""
from __future__ import annotations

import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera_output import frame_converter as conv      # noqa: E402
from camera_output import roycam_format as F           # noqa: E402
from camera_output.ring import RingReader, RingWriter  # noqa: E402


def _gradient(w=320, h=240):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    xs = np.linspace(0, 255, w, dtype=np.uint8)
    ys = np.linspace(0, 255, h, dtype=np.uint8)
    img[:, :, 0] = xs[None, :]
    img[:, :, 1] = ys[:, None]
    img[:, :, 2] = ((xs[None, :].astype(int) + ys[:, None].astype(int)) // 2)
    return img


def test_contract_sizes():
    assert F.CONTROL_SZ == 64
    assert F.HEADER_SZ == 128
    c = F.pack_control(100, 1920, 1080, 1, F.SRC_ACTIVE)
    assert len(c) == F.CONTROL_SZ
    h = F.pack_header(F.FMT_NV12, 1280, 720, 1280, 100, 30, 1, 7, 123, 2,
                      F.FLAG_READY)
    assert len(h) == F.HEADER_SZ
    assert F.unpack_header(h)["width"] == 1280
    print("OK contract sizes")


def _roundtrip_mae(fmt_id):
    src = _gradient()
    payload, w, h, stride = conv.convert(src, fmt_id)
    assert len(payload) == F.payload_size(fmt_id, w, h), "payload size mismatch"
    assert stride == F.stride_for(fmt_id, w)
    back = conv.decode(payload, fmt_id, w, h)
    mae = np.mean(np.abs(src[:h, :w].astype(int) - back.astype(int)))
    return mae


def test_roundtrip_nv12():
    # 2x2 chroma subsampling + BT.601 round-trip on a steep gradient; real
    # camera content is far smoother. Bound guards against gross corruption.
    mae = _roundtrip_mae(F.FMT_NV12)
    assert mae < 12.0, f"NV12 MAE too high: {mae}"
    print(f"OK NV12 round-trip MAE={mae:.2f}")


def test_roundtrip_yuy2():
    mae = _roundtrip_mae(F.FMT_YUY2)
    assert mae < 12.0, f"YUY2 MAE too high: {mae}"
    print(f"OK YUY2 round-trip MAE={mae:.2f}")


def test_roundtrip_rgb32():
    mae = _roundtrip_mae(F.FMT_RGB32)
    assert mae < 0.01, f"RGB32 must be lossless, MAE={mae}"
    print(f"OK RGB32 round-trip MAE={mae:.4f}")


def test_ring_roundtrip(tmp_path=None):
    path = os.path.join(os.environ.get("TEMP", "."), "roycam_test_ring.bin")
    w = RingWriter(640, 480, F.FMT_NV12, path=path)
    try:
        src = _gradient(320, 240)
        payload, fw, fh, _ = conv.convert(src, F.FMT_NV12)
        w.write(payload, F.FMT_NV12, fw, fh)
        r = RingReader(path)
        got = r.read_latest()
        assert got is not None, "reader saw nothing"
        header, data = got
        assert header["width"] == fw and header["height"] == fh
        assert data == payload, "payload mismatch through ring"
        r.close()
        print("OK ring round-trip")
    finally:
        w.close()


def test_ring_tearfree_30fps(tmp_path=None):
    """Writer pushes distinct frames at 30fps; a concurrent reader must never
    observe a torn frame (mixed content / inconsistent seq)."""
    path = os.path.join(os.environ.get("TEMP", "."), "roycam_tear_ring.bin")
    width, height = 320, 240
    w = RingWriter(640, 480, F.FMT_RGB32, path=path)
    stop = threading.Event()
    torn = {"count": 0, "reads": 0}

    def writer():
        n = 0
        while not stop.is_set():
            # Solid color keyed to frame number -> easy tear detection.
            val = n % 250
            img = np.full((height, width, 3), val, dtype=np.uint8)
            payload, fw, fh, _ = conv.convert(img, F.FMT_RGB32)
            w.write(payload, F.FMT_RGB32, fw, fh)
            n += 1
            time.sleep(1.0 / 30.0)

    def reader():
        r = RingReader(path)
        while not stop.is_set():
            got = r.read_latest()
            if got is None:
                continue
            header, data = got
            arr = np.frombuffer(data, dtype=np.uint8)
            torn["reads"] += 1
            # A clean RGB32 solid frame: every BGRA pixel shares one B/G/R value.
            b = arr[0::4]
            if not (np.all(b == b[0])):
                torn["count"] += 1
        r.close()

    tw = threading.Thread(target=writer, daemon=True)
    tr = threading.Thread(target=reader, daemon=True)
    tw.start()
    tr.start()
    time.sleep(2.0)
    stop.set()
    tw.join(1.0)
    tr.join(1.0)
    w.close()
    assert torn["reads"] > 10, f"reader barely ran ({torn['reads']} reads)"
    assert torn["count"] == 0, f"{torn['count']} torn frames in {torn['reads']}"
    print(f"OK tear-free: {torn['reads']} reads, 0 torn")


if __name__ == "__main__":
    test_contract_sizes()
    test_roundtrip_nv12()
    test_roundtrip_yuy2()
    test_roundtrip_rgb32()
    test_ring_roundtrip()
    test_ring_tearfree_30fps()
    print("\nALL PHASE-1 TESTS PASSED")

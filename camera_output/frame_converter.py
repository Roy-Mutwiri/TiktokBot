"""
frame_converter.py — convert BGR frames (OpenCV native) to the camera's
advertised pixel formats. ALL conversion happens here in user mode; the
service/driver only copy pre-formatted bytes.

Formats:
  NV12  (primary)  : Y plane + interleaved UV, 12 bpp
  YUY2  (secondary): packed 4:2:2 (YUYV byte order), 16 bpp
  RGB32 (debug)    : BGRA top-down, 32 bpp

Width/height are forced even (4:2:2 / 4:2:0 require it). Returns the raw
payload bytes plus the (width, height, stride) actually produced.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - cv2 is a hard dep in this repo
    cv2 = None

from . import roycam_format as fmt


def _even(frame: np.ndarray) -> np.ndarray:
    """Crop to even width/height (chroma subsampling needs it)."""
    h, w = frame.shape[:2]
    h2, w2 = h - (h & 1), w - (w & 1)
    if (h2, w2) != (h, w):
        frame = frame[:h2, :w2]
    return frame


def bgr_to_nv12(bgr: np.ndarray) -> bytes:
    bgr = _even(np.ascontiguousarray(bgr))
    h, w = bgr.shape[:2]
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
    y = yuv[:, :, 0]
    u = yuv[::2, ::2, 1]      # subsample 2x2 -> (h/2, w/2)
    v = yuv[::2, ::2, 2]
    out = np.empty((h * 3 // 2, w), dtype=np.uint8)
    out[:h] = y
    uv = out[h:].reshape(h // 2, w)
    uv[:, 0::2] = u           # interleaved U,V
    uv[:, 1::2] = v
    return out.tobytes()


def bgr_to_yuy2(bgr: np.ndarray) -> bytes:
    bgr = _even(np.ascontiguousarray(bgr))
    h, w = bgr.shape[:2]
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
    y = yuv[:, :, 0]
    u = yuv[:, 0::2, 1]       # one chroma sample per 2 px horizontally
    v = yuv[:, 0::2, 2]
    out = np.empty((h, w * 2), dtype=np.uint8)
    out[:, 0::2] = y          # Y0 Y1 Y2 ... at even byte positions
    out[:, 1::4] = u          # U at positions 1,5,9,...
    out[:, 3::4] = v          # V at positions 3,7,11,...
    return out.tobytes()


def bgr_to_rgb32(bgr: np.ndarray) -> bytes:
    bgr = _even(np.ascontiguousarray(bgr))
    bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)   # Windows RGB32 == BGRA
    return bgra.tobytes()


def convert(bgr: np.ndarray, fmt_id: int) -> Tuple[bytes, int, int, int]:
    """Return (payload_bytes, width, height, stride) for the chosen format."""
    bgr = _even(np.ascontiguousarray(bgr))
    h, w = bgr.shape[:2]
    if fmt_id == fmt.FMT_NV12:
        data = bgr_to_nv12(bgr)
    elif fmt_id == fmt.FMT_YUY2:
        data = bgr_to_yuy2(bgr)
    elif fmt_id == fmt.FMT_RGB32:
        data = bgr_to_rgb32(bgr)
    else:
        raise ValueError(f"unsupported format id {fmt_id}")
    return data, w, h, fmt.stride_for(fmt_id, w)


# --- decode helpers (used by round-trip tests / the ring reader preview) ------

def nv12_to_bgr(data: bytes, w: int, h: int) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8).reshape(h * 3 // 2, w)
    return cv2.cvtColor(arr, cv2.COLOR_YUV2BGR_NV12)


def yuy2_to_bgr(data: bytes, w: int, h: int) -> np.ndarray:
    # YUY2 decode expects a 2-channel image (Y, packed-chroma) per pixel.
    arr = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 2)
    return cv2.cvtColor(arr, cv2.COLOR_YUV2BGR_YUY2)


def rgb32_to_bgr(data: bytes, w: int, h: int) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 4)
    return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)


def decode(data: bytes, fmt_id: int, w: int, h: int) -> np.ndarray:
    if fmt_id == fmt.FMT_NV12:
        return nv12_to_bgr(data, w, h)
    if fmt_id == fmt.FMT_YUY2:
        return yuy2_to_bgr(data, w, h)
    if fmt_id == fmt.FMT_RGB32:
        return rgb32_to_bgr(data, w, h)
    raise ValueError(f"unsupported format id {fmt_id}")

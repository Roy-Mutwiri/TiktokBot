"""
roycam_format.py — Python mirror of shared/roycam_frame_format.h (v2).

Single source of truth for the shared-memory layout, kept byte-for-byte
identical to the C header so the future service/driver read the same bytes.
Layout:  [ RoycControl 64B ] [ slot0 ][ slot1 ][ slot2 ]
         where each slot = [ RoycFrameHeader 128B ][ payload ... ]

This is a SOFTWARE camera contract. It carries app-generated frames; it does
not describe physical USB hardware.
"""
from __future__ import annotations

import struct

MAGIC        = 0x43594F52   # 'ROYC' little-endian
VERSION      = 2
HEADER_SZ    = 128          # per-slot frame header
CONTROL_SZ   = 64           # control block at offset 0
RING_SLOTS   = 3            # triple buffer

# Pixel formats (must match RoycPixelFormat).
FMT_NV12  = 1
FMT_YUY2  = 2
FMT_RGB32 = 3

FORMAT_NAMES = {FMT_NV12: "NV12", FMT_YUY2: "YUY2", FMT_RGB32: "RGB32"}

# Source states (RoycSourceState).
SRC_NONE   = 0
SRC_ACTIVE = 1
SRC_PAUSED = 2

# status_flags bits.
FLAG_READY    = 0x1
FLAG_FALLBACK = 0x2

# Status codes (mirror shared/roycam_status_codes.h).
ST_OK                   = 0
ST_DRIVER_NOT_INSTALLED = 1
ST_SERVICE_STOPPED      = 2
ST_NO_SOURCE            = 3
ST_FRAME_TOO_LARGE      = 4
ST_UNSUPPORTED_FORMAT   = 5
ST_RESOLUTION_MISMATCH  = 6
ST_IPC_TIMEOUT          = 7
ST_CLIENT_BUSY          = 8
ST_FALLBACK_ACTIVE      = 9
ST_BAD_HEADER           = 10

# --- struct layouts (little-endian, packed) ----------------------------------
# RoycControl: 9 u32 + 7 u32 reserved = 16 u32 = 64 bytes.
_CONTROL_FMT = "<16I"
# RoycFrameHeader: 8 u32 + 3 u64 + 2 u32 + 16 u32 reserved = 128 bytes.
_HEADER_FMT  = "<8I3Q2I16I"

assert struct.calcsize(_CONTROL_FMT) == CONTROL_SZ, "control size drift"
assert struct.calcsize(_HEADER_FMT) == HEADER_SZ, "header size drift"


def payload_size(fmt: int, w: int, h: int) -> int:
    """Bytes of pixel payload for a format/resolution (RoycPayloadSize)."""
    px = w * h
    if fmt == FMT_NV12:
        return px + px // 2          # 12 bpp
    if fmt == FMT_YUY2:
        return px * 2                # 16 bpp
    if fmt == FMT_RGB32:
        return px * 4                # 32 bpp
    return 0


def stride_for(fmt: int, w: int) -> int:
    """Bytes per row of plane 0."""
    if fmt == FMT_NV12:
        return w                     # Y plane row
    if fmt == FMT_YUY2:
        return w * 2
    if fmt == FMT_RGB32:
        return w * 4
    return 0


def pack_control(slot_stride: int, max_w: int, max_h: int,
                 write_index: int, source_state: int) -> bytes:
    return struct.pack(
        _CONTROL_FMT,
        MAGIC, VERSION, HEADER_SZ, RING_SLOTS, slot_stride,
        max_w, max_h, write_index & 0xFFFFFFFF, source_state,
        0, 0, 0, 0, 0, 0, 0,
    )


def unpack_control(buf: bytes):
    v = struct.unpack(_CONTROL_FMT, buf[:CONTROL_SZ])
    return {
        "magic": v[0], "version": v[1], "header_size": v[2],
        "ring_slots": v[3], "slot_stride": v[4],
        "max_width": v[5], "max_height": v[6],
        "write_index": v[7], "source_state": v[8],
    }


def pack_header(fmt: int, w: int, h: int, stride: int, data_size: int,
                fps_num: int, fps_den: int, frame_index: int,
                timestamp_qpc: int, seq: int, status_flags: int,
                checksum: int = 0) -> bytes:
    return struct.pack(
        _HEADER_FMT,
        MAGIC, fmt, w, h, stride, data_size, fps_num, fps_den,
        frame_index & 0xFFFFFFFFFFFFFFFF,
        timestamp_qpc & 0xFFFFFFFFFFFFFFFF,
        seq & 0xFFFFFFFFFFFFFFFF,
        status_flags, checksum,
        *([0] * 16),
    )


def unpack_header(buf: bytes):
    v = struct.unpack(_HEADER_FMT, buf[:HEADER_SZ])
    return {
        "magic": v[0], "format": v[1], "width": v[2], "height": v[3],
        "stride": v[4], "data_size": v[5], "fps_num": v[6], "fps_den": v[7],
        "frame_index": v[8], "timestamp_qpc": v[9], "seq": v[10],
        "status_flags": v[11], "checksum": v[12],
    }


def total_map_size(slot_stride: int) -> int:
    return CONTROL_SZ + RING_SLOTS * slot_stride


def slot_offset(slot_stride: int, slot: int) -> int:
    return CONTROL_SZ + slot * slot_stride

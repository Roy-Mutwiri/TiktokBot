# =============================================================================
# avatar_sharedframe.py  —  shared-memory bridge to the native virtual camera
# -----------------------------------------------------------------------------
# The native Media Foundation virtual camera (native_camera\) runs INSIDE the
# Windows Camera Frame Server process. It can't reach into this Python process,
# so we hand frames across via a small memory-mapped FILE that both sides open
# by the same well-known path. A file (not a Global\ pagefile section) is used
# deliberately: creating Global\ shared memory needs admin privilege, but a file
# under C:\Users\Public is readable by the Frame Server with no special rights
# and works even when it runs in a different session.
#
# Layout (little-endian):
#   offset 0  : u32 magic   = 0x31435641  ('AVC1')
#   offset 4  : u32 version = 1
#   offset 8  : u32 width
#   offset 12 : u32 height
#   offset 16 : u32 fourcc  = 0 (RGB32 / BGRA, top-down)
#   offset 24 : u64 seq     (seqlock: odd = write in progress, even = stable)
#   offset 64 : frame bytes, width*height*4  (B,G,R,A per pixel)
#
# The native reader uses the SAME constants (see native_camera\SharedFrame.h).
# =============================================================================

import os
import mmap
import struct

MAGIC = 0x31435641           # 'AVC1'
VERSION = 1
HEADER_SIZE = 64
SEQ_OFFSET = 24
# Both sides agree on this path. C:\Users\Public is world-accessible, so the
# Frame Server (even in another session/account) can open it read-only.
DEFAULT_PATH = os.path.join(
    os.environ.get("PUBLIC", r"C:\Users\Public"), "AvatarStudioCamera", "frame.bin")


def buffer_size(width, height):
    return HEADER_SIZE + width * height * 4


class SharedFrameWriter:
    """Publishes BGR frames into the shared file the native camera reads."""

    def __init__(self, width=512, height=512, path=None):
        self.width = int(width)
        self.height = int(height)
        self.path = path or DEFAULT_PATH
        self._seq = 0
        size = buffer_size(self.width, self.height)

        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        # Create / size the backing file to exactly `size` bytes.
        if not os.path.exists(self.path) or os.path.getsize(self.path) != size:
            with open(self.path, "wb") as f:
                f.truncate(size)

        self._f = open(self.path, "r+b")
        self._mm = mmap.mmap(self._f.fileno(), size)
        # Write a stable header up front (seq stays even until first frame).
        struct.pack_into("<IIIII", self._mm, 0,
                         MAGIC, VERSION, self.width, self.height, 0)
        self._set_seq(0)

    def _set_seq(self, value):
        struct.pack_into("<Q", self._mm, SEQ_OFFSET, value & 0xFFFFFFFFFFFFFFFF)

    def write(self, bgr):
        """Push one frame. `bgr` is an HxWx3 uint8 BGR ndarray (any size)."""
        import numpy as np
        import cv2
        if bgr.shape[1] != self.width or bgr.shape[0] != self.height:
            bgr = cv2.resize(bgr, (self.width, self.height))
        # BGR -> BGRA (RGB32 is B,G,R,A in memory), top-down.
        bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        data = np.ascontiguousarray(bgra).tobytes()

        # seqlock: odd before, even after, so the reader never sees a half-frame.
        self._seq += 1
        self._set_seq(self._seq * 2 - 1)          # odd
        self._mm[HEADER_SIZE:HEADER_SIZE + len(data)] = data
        self._set_seq(self._seq * 2)              # even
        # mmap writes are visible to other openers without an explicit flush;
        # flush opportunistically so a file-system observer also sees progress.

    def close(self):
        try:
            self._mm.flush()
            self._mm.close()
        except Exception:
            pass
        try:
            self._f.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class SharedFrameReader:
    """Reference reader (mirrors the native C++ reader) — for tests/diagnostics."""

    def __init__(self, path=None):
        self.path = path or DEFAULT_PATH
        self._f = open(self.path, "r+b")
        size = os.path.getsize(self.path)
        self._mm = mmap.mmap(self._f.fileno(), size)

    def header(self):
        magic, ver, w, h, fourcc = struct.unpack_from("<IIIII", self._mm, 0)
        seq, = struct.unpack_from("<Q", self._mm, SEQ_OFFSET)
        return dict(magic=magic, version=ver, width=w, height=h,
                   fourcc=fourcc, seq=seq)

    def read(self):
        """Return (seq, HxWx3 BGR ndarray) using the seqlock, or (seq, None)."""
        import numpy as np
        import cv2
        for _ in range(8):
            s1, = struct.unpack_from("<Q", self._mm, SEQ_OFFSET)
            if s1 == 0 or (s1 & 1):
                continue
            _, _, w, h, _ = struct.unpack_from("<IIIII", self._mm, 0)
            n = w * h * 4
            buf = bytes(self._mm[HEADER_SIZE:HEADER_SIZE + n])
            s2, = struct.unpack_from("<Q", self._mm, SEQ_OFFSET)
            if s1 == s2:
                bgra = np.frombuffer(buf, np.uint8).reshape(h, w, 4)
                return s1 // 2, cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        return 0, None

    def close(self):
        try:
            self._mm.close(); self._f.close()
        except Exception:
            pass

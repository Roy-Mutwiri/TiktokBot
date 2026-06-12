# IPC & Frame Transport

How frames move **App → Service → Driver**, the wire format, the ring-buffer
design, validation rules, and fallback behavior.

---

## 1. Transport choice (A–G compared)

| Option | Use here? | Why |
|--------|-----------|-----|
| **A. Shared-memory ring buffer** | **App ↔ Service: YES (primary)** | Lowest latency, ample bandwidth for 1080p30 raw, no syscall per frame. |
| B. Named pipe | Control/status only | Easy, but per-message overhead is wrong for bulk raw frames. Great for start/stop/status. |
| C. IOCTL transfer | **Service ↔ Driver: YES (primary)** | Standard, well-defined user↔kernel boundary with explicit buffer validation. |
| D. MMF via service intermediary | **App ↔ Service: YES** (this *is* A, backed by a memory-mapped file) | Current `frame.bin` already uses an MMF; we keep that mechanism, add a ring + v2 header. |
| E. Driver-owned buffer mapped to user mode | Considered, **not first** | Mapping kernel buffers to user mode adds attack surface + complexity; defer. |
| F. Localhost socket / gRPC | Service-only optional | Fine for remote/diagnostic control; not for driver frame path. |
| G. Hybrid service + shared memory | **This is the chosen shape** | App↔Service = shared-memory ring (A/D); Service↔Driver = IOCTL (C). |

**Decision:**
- **App → Service:** memory-mapped **triple-buffer ring** (evolution of today's
  single-frame `C:\Users\Public\AvatarStudioCamera\frame.bin`).
- **Service → Driver:** **IOCTL** frame-submit + control, with a validated boundary.
  (Optionally a service-mapped driver section later for zero-copy; not in the MVP.)
- **Control/status:** **named pipe** (small JSON/struct messages), never frame data.

This keeps the driver simple and the kernel/user boundary auditable.

---

## 2. Frame header v2 (`roycam_frame_format.h`)

Single source of truth in `/shared`, mirrored to the driver's `public/` and the
Python client. Little-endian, fixed 128-byte header, then the payload.

```c
// roycam_frame_format.h  — shared by app (Python ctypes), service, driver
#pragma once
#include <stdint.h>

#define ROYC_MAGIC      0x43594F52u   /* 'R''O''Y''C' little-endian */
#define ROYC_VERSION    2
#define ROYC_HEADER_SZ  128
#define ROYC_RING_SLOTS 3             /* triple buffer */

typedef enum {
    ROYC_FMT_NV12  = 1,   /* primary  (Y plane + interleaved UV, 12bpp) */
    ROYC_FMT_YUY2  = 2,   /* secondary(packed 4:2:2, 16bpp)             */
    ROYC_FMT_RGB32 = 3,   /* debug    (BGRA top-down, 32bpp)            */
} RoycPixelFormat;

typedef enum {
    ROYC_SRC_NONE   = 0,  /* no source active -> fallback               */
    ROYC_SRC_ACTIVE = 1,  /* app producing frames                       */
    ROYC_SRC_PAUSED = 2,
} RoycSourceState;

/* Control block at the very start of the shared section (offset 0).      */
typedef struct {
    uint32_t magic;            /* ROYC_MAGIC                              */
    uint32_t version;          /* ROYC_VERSION                           */
    uint32_t header_size;      /* ROYC_HEADER_SZ                         */
    uint32_t ring_slots;       /* ROYC_RING_SLOTS                        */
    uint32_t slot_stride;      /* bytes per slot = ROYC_HEADER_SZ + max payload */
    uint32_t max_width;        /* capacity (e.g. 1920)                   */
    uint32_t max_height;       /* capacity (e.g. 1080)                   */
    volatile uint32_t write_index;  /* atomic: last fully-written slot   */
    volatile uint32_t source_state; /* RoycSourceState                   */
    uint32_t reserved[7];      /* pad control block to 64 bytes          */
} RoycControl;                 /* sizeof == 64                            */

/* Per-slot frame header (precedes each slot's payload).                  */
typedef struct {
    uint32_t magic;            /* ROYC_MAGIC (per-slot sanity)           */
    uint32_t format;           /* RoycPixelFormat                        */
    uint32_t width;            /* active frame width                     */
    uint32_t height;           /* active frame height                    */
    uint32_t stride;           /* bytes per row of plane 0               */
    uint32_t data_size;        /* payload bytes actually written         */
    uint32_t fps_num;          /* e.g. 30                                */
    uint32_t fps_den;          /* e.g. 1                                 */
    uint64_t frame_index;      /* monotonic                              */
    uint64_t timestamp_qpc;    /* QueryPerformanceCounter at write       */
    volatile uint64_t seq;     /* SEQLOCK: odd=writing, even=stable      */
    uint32_t status_flags;     /* bit0=key/ready, ...                    */
    uint32_t checksum;         /* optional CRC32 of payload (0 = skip)   */
    uint32_t reserved[8];      /* pad header to 128 bytes                */
} RoycFrameHeader;             /* sizeof == 128 (ROYC_HEADER_SZ)         */
```

**Migration from v1 (`AVC1`, today's `avatar_sharedframe.py`):** the v1 buffer is a
single 64-byte header + one BGRA frame guarded by a seqlock at offset 24. v2 keeps the
**same seqlock idea per slot** but adds the control block, ring, format/stride/fps/QPC,
and `source_state`. The Python writer will support **both** formats during migration so
the existing MF/DShow cameras keep reading v1 until they're switched to v2.

## 3. Ring buffer (App → Service)

- **Triple buffer** (`ROYC_RING_SLOTS = 3`): writer never waits on a reader; reader
  always gets the newest complete frame; latest-frame-wins.
- **Write (app):**
  1. pick `slot = (control.write_index + 1) % slots`,
  2. `seq++` on that slot's header (now **odd** → "writing"),
  3. write header fields + copy payload,
  4. `seq++` (now **even** → "stable"),
  5. publish: `InterlockedExchange(&control.write_index, slot)` (release).
- **Read (service):**
  1. `slot = control.write_index` (acquire),
  2. read `seq` (retry if odd or changed across the copy) → tear-free,
  3. validate header, copy/forward.
- No locks needed for the data path; only atomics + the per-slot seqlock. The MMF is
  sized `64 + slots * (128 + max_width*max_height*4)` (RGB32 worst case ≈ 24 MB for
  3×1080p — acceptable for an MMF; NV12/YUY2 are smaller).

## 4. Service → Driver (IOCTL)

`/shared/roycam_ioctl.h` defines a tiny, validated control surface:

```c
#define ROYC_IOCTL_SUBMIT_FRAME  CTL_CODE(FILE_DEVICE_UNKNOWN, 0x800, METHOD_IN_DIRECT,  FILE_WRITE_ACCESS)
#define ROYC_IOCTL_SET_FORMAT    CTL_CODE(FILE_DEVICE_UNKNOWN, 0x801, METHOD_BUFFERED,   FILE_WRITE_ACCESS)
#define ROYC_IOCTL_QUERY_STATUS  CTL_CODE(FILE_DEVICE_UNKNOWN, 0x802, METHOD_BUFFERED,   FILE_READ_ACCESS)
```

- **SUBMIT_FRAME** carries a `RoycFrameHeader` + payload (METHOD_IN_DIRECT so the
  large payload is mapped, not double-copied). Driver validates *everything* before
  touching a byte (see §6) and copies into its **preallocated** nonpaged latest-frame
  buffer under a short lock.
- **SET_FORMAT / QUERY_STATUS** are METHOD_BUFFERED (small, simple). Status returns
  fps, last-frame age, client-open count, error counters.
- The driver **owns its own output timing** and never blocks waiting for SUBMIT.

## 5. Fallback behavior

- `source_state == ROYC_SRC_NONE`, **or** newest frame older than `T_stale`
  (default 2 frame intervals → repeat last frame; > `T_fallback` ≈ 1 s → fallback
  frame), **or** app/service gone → serve the **branded fallback**:
  *"RoyCam HD Camera — waiting for source"*.
- Fallback frame is **preconverted** to each supported format and cached (service
  generates it; driver also embeds a minimal internal pattern so it's safe even with
  no service at all).

## 6. Validation rules (driver treats user mode as hostile)

Before accepting any SUBMIT_FRAME, the driver checks, with **overflow-safe math**:
- `header.magic == ROYC_MAGIC`, `format ∈ {NV12,YUY2,RGB32}`.
- `width,height` within `[min, max]` and **match the negotiated media type** for the
  open pin (reject mid-stream resolution changes that the pin didn't agree to).
- `stride >= bytes_per_row(format,width)` and `stride` row-aligned.
- `data_size == expected_size(format,width,height,stride)` computed via checked
  multiply (`RtlULongLongMult`-style) — reject on overflow or mismatch.
- `InputBufferLength >= ROYC_HEADER_SZ + data_size`.
- Copy is bounded to `min(data_size, preallocated_capacity)`; never trust the caller's
  length alone. No allocation in the hot path; no blocking; lock held only for the copy.

Service-side mirrors these checks before pushing, and additionally **rejects
unsupported sizes/formats** so bad frames never reach the kernel.

## 7. Stress / correctness tests (deliverables)

- `tests/shared-memory-tests`: writer + reader, 30fps for **10 minutes**, assert zero
  tears (seqlock), zero header-magic failures, monotonic frame index, latest-wins.
- Fault injection: kill writer mid-frame → reader retries cleanly; oversized/garbage
  header → reader/driver rejects without UB.
- Throughput: sustain 1080p30 NV12 (~46 MB/s) with bounded CPU; measure p99 write→read
  latency (target < 1 frame interval).
- 6-hour endurance: no MMF growth, no handle leak, no kernel nonpaged-pool growth.

## 8. Why this is safe & maintainable
- One header file (`/shared`) defines the contract for all three layers → no drift.
- Kernel only ever does **bounded copies of already-formatted frames**; all
  conversion/validation heavy-lifting is user mode.
- The same ring feeds **either** the user-mode software camera (track A, today) **or**
  the AVStream driver (track B) — we can develop and ship incrementally.

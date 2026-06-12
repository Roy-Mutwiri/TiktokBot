# Driver Design — RoyCam AVStream Camera Minidriver

Design of the kernel-mode camera **device** ("RoyCam HD Camera"). This is the
"track B" professional driver. It is a **software-backed AVStream camera** — we
say so honestly everywhere; we never fake USB hardware, never spoof IDs/signatures,
never impersonate a vendor.

**Status:** design only. No skeleton code until the WDK is installed, dev
test-signing is enabled, and there is an explicit go-ahead.

---

## 1. Model choice: AVStream minidriver, AvsCamera as the conceptual base

- **AVStream (KS)** is the Windows kernel-streaming model for camera/capture
  devices. It plugs into the modern camera stack (Frame Server → Media Foundation →
  DirectShow bridge), so a correctly-built AVStream camera is visible to the Windows
  Camera app, Chrome/WebRTC, Zoom/Teams, TikTok, etc.
- Microsoft's **AvsCamera** sample is a *simulated* AVStream camera: a `CCaptureDevice`
  + `CCapturePin` + a `CHardwareSimulation` that synthesizes frames. It already
  demonstrates capture pins, KS media formats, frame delivery, and clean start/stop.
  **We use it as the conceptual base** and swap the simulation for our app-fed frame
  source. We do **not** copy it blindly — we re-implement with our branding,
  hardened validation, and our IPC.

> Honesty note: the alternative user-mode `MFCreateVirtualCamera` source (already
> built, `native_camera/`) is Microsoft's *sanctioned* route for software frames and
> is what ships today. AVStream is the heavier "real camera device" path; we pursue
> it deliberately with the signing reality understood (`driver-build-and-signing.md`).

## 2. Device & PnP model

| Item | Value |
|------|-------|
| Bus / enumeration | **Root-enumerated software device** (no fake USB bus). Hardware ID in **our own** namespace, e.g. `Root\RoyCamHDCamera`. |
| Device class | `Camera` / `Image`; categories advertised: **`KSCATEGORY_VIDEO_CAMERA`**, `KSCATEGORY_CAPTURE`, `KSCATEGORY_VIDEO` (so the Frame Server and DShow bridge enumerate it). |
| Friendly name | **"RoyCam HD Camera"** (configurable string in INF). |
| Manufacturer / Provider | **Our brand** (e.g. "RoyCam"). Never another vendor. |
| Driver type | AVStream minidriver (`ks.sys` client), Universal-driver-compatible target where feasible. |

The INF installs one software camera device. No physical-bus claims, no UVC
descriptors, no third-party IDs.

## 3. Filter / pin topology

```
RoyCam filter (KS filter factory)
└─ Capture pin (PINDIR_OUTPUT, KSPIN_COMMUNICATION_BOTH, category PIN_CATEGORY_CAPTURE)
     media types (priority order):
       1. NV12  1920x1080 @ 30   (KSDATAFORMAT_SUBTYPE_NV12, KS_VIDEOINFOHEADER2)
       2. NV12  1280x720  @ 30
       3. YUY2  1280x720  @ 30   (KSDATAFORMAT_SUBTYPE_YUY2)
       4. YUY2  640x480   @ 30
       5. (RGB32 640x480 @ 30  — DEBUG build only)
```

- One output capture pin in v1 (single-stream). Multi-pin/preview+capture can come
  later; v1 documents single-client behavior.
- Each media type is a `KSDATARANGE_VIDEO` (or `_VIDEO2`) entry in
  `media_formats.*`, with correct `bmiHeader` (biWidth/biHeight/biBitCount/
  biCompression), `AvgTimePerFrame = 333333` (30fps), and min/max frame size.
- Default/preferred type = NV12 1280×720 @ 30.

## 4. Frame source — three staged implementations (per the brief)

The driver's frame source is abstracted behind an internal `IFrameProvider`
(kernel-side) so we can swap it without touching the pin/streaming code:

1. **Stage A — simulated:** internal test pattern (moving color bars + frame
   counter + timestamp). Proves the camera works in the stack end-to-end with our
   name. (Mirrors AvsCamera's simulation.)
2. **Stage B — static:** one fixed test frame (from a resource or default buffer).
   Proves format/stride/delivery without external input.
3. **Stage C — service-fed:** the **latest validated frame** pushed by the service
   over IOCTL (see `ipc-frame-transport.md`). Falls back to Stage A's pattern when no
   frame/service is present.

The pin always works; the provider just changes where the bytes come from.

## 5. Frame buffer & delivery (the hot path)

- **Preallocated nonpaged buffers** sized for the largest media type (1080p NV12),
  allocated at **pin start (state→Acquire/Run)**, freed at **pin stop**. **No
  per-frame allocation.**
- A **double latest-frame buffer** inside the driver: the IOCTL writer fills the
  back buffer and flips an index (atomic); the streaming path reads the front buffer.
  A lightweight spinlock guards only the index flip and the bounded copy.
- **Streaming callback** (`CCapturePin::Process` / KS frame delivery):
  - never blocks, never does I/O, never touches pageable memory at DISPATCH_LEVEL;
  - copies `min(frame_size, sample_capacity)` bytes from the front buffer into the KS
    stream sample;
  - sets the sample's PTS/duration from the **driver's own pacing clock**, marks it a
    sync point.
- **Pacing:** an internal timer/DPC (or KS clock) drives 30fps regardless of how often
  the service pushes. If the latest frame is older than `T_stale` → repeat last; older
  than `T_fallback` → internal fallback pattern (Stage A). This guarantees a smooth,
  non-frozen feed even if the app stutters or dies.

## 6. IOCTL surface (kernel/user boundary — validate everything)

From `public/roycam_ioctl.h` (mirrors `/shared/roycam_ioctl.h`):

| IOCTL | Method | Purpose | Validation |
|-------|--------|---------|------------|
| `ROYC_IOCTL_SET_FORMAT` | BUFFERED | service announces format it will send | format ∈ enum; W/H within caps; matches an advertised media type |
| `ROYC_IOCTL_SUBMIT_FRAME` | IN_DIRECT | push latest frame (header + payload) | magic, format, W/H = negotiated type, stride ≥ row bytes, `data_size` via checked-multiply, `InputBufferLength ≥ header+data_size`; bounded copy |
| `ROYC_IOCTL_QUERY_STATUS` | BUFFERED | fps, last-frame age, client-open count, errors | output buffer length check |

**Rules (enforced in code review):** verify `Irp` method matches the IOCTL; validate
`InputBufferLength`/`OutputBufferLength`; never trust caller lengths; overflow-safe
math (`RtlULongLongMult`/`RtlSizeTMult`); copy bounded to preallocated capacity; no
allocation/blocking on the submit path beyond the short lock. Reject (STATUS_INVALID_*)
rather than risk a bugcheck.

## 7. Start/stop, power, and state

- KS pin state machine: Stop → Acquire (allocate buffers, start pacing) → Pause →
  Run → … → Stop (stop pacing, free buffers). Idempotent and leak-free across cycles.
- PnP/power: handle `IRP_MN_START_DEVICE`/`REMOVE`/`SURPRISE_REMOVAL`; on
  Dx/sleep, stop pacing and release; on resume, re-arm. No crash on surprise removal
  of the (virtual) device. Test sleep/wake (testing matrix #12).

## 8. File layout (`/driver/RoyCamDriver/`)

```
RoyCamDriver.sln / .vcxproj         # WDK driver project (Universal target)
RoyCam.inx -> RoyCam.inf            # stampinf-generated INF
RoyCam.rc                           # version/branding resource (our strings only)
device.c/.h        # CCaptureDevice: PnP, filter factory, lifetime
capture.c/.h       # CCapturePin: media types, state, frame delivery, pacing
frame_buffer.c/.h  # preallocated double buffer, latest-frame, fallback pattern
ioctl.c/.h         # IOCTL dispatch + validation (the trust boundary)
media_formats.c/.h # KSDATARANGE_VIDEO table (NV12/YUY2/RGB32)
trace.h            # WPP/ETW tracing (no user content)
public/
  roycam_ioctl.h   roycam_public.h   # mirrors /shared
```

Strong comments at every kernel/user boundary; all conversion stays user-mode.

## 9. INF design (branding + honesty)

- `[Version]`: our `Provider`, `CatalogFile = RoyCam.cat`, `Class = Camera`,
  `ClassGuid = {ca3e7ab9-b4c3-4ae6-8251-579ef933890f}` (Camera class).
- `[Strings]`: `RoyCamHDCamera.DeviceDesc = "RoyCam HD Camera"`, manufacturer = our
  brand. **No third-party names.**
- Adds the device to `KSCATEGORY_VIDEO_CAMERA`/`KSCATEGORY_CAPTURE`/`KSCATEGORY_VIDEO`
  interfaces so the Frame Server enumerates it.
- A comment header states plainly: *software-backed camera; not a physical USB device.*

## 10. Build & dev-signing (summary; full in `driver-build-and-signing.md`)

- Build with the **WDK 10.0.26100** (matches our SDK) via msbuild:
  `msbuild RoyCamDriver.vcxproj /p:Configuration=Release /p:Platform=x64`.
- Dev: generate a **self-signed test cert**, sign `.sys`+`.cat`, enable
  `bcdedit /set testsigning on` (**dev machine only**, watermark), install with
  `pnputil /add-driver RoyCam.inf /install`.
- Production: EV cert + Microsoft Partner Center attestation/WHQL. **Unsigned/
  test-signed drivers do not load on normal machines** — this is the distribution gate.

## 11. Driver-specific risks

| Risk | Mitigation |
|------|-----------|
| Bugcheck from bad user input | Strict IOCTL validation, overflow-safe math, bounded copies (§6). |
| Stutter / frozen feed | Driver owns 30fps pacing + repeat-last + fallback (§5). |
| Nonpaged-pool growth / leaks | Preallocate at start, free at stop; endurance test (matrix #15/16). |
| Won't load (signing) | Honest signing plan; user-mode camera remains the shipping fallback. |
| Frame Server doesn't enumerate it | Correct KSCATEGORY interfaces + INF; verify in Camera app first (matrix #2/3). |
| Multi-client open | v1 documents single-client; reject 2nd open gracefully; revisit later. |

## 12. Internal build phases (do not skip)
1. WDK installed; empty driver builds + loads (Device Manager shows "RoyCam HD Camera").
2. Capture pin + media formats; Windows Camera app opens it; **Stage A** pattern.
3. **Stage B** static frame; verify stride/format correctness across formats.
4. IOCTL surface + validation; **Stage C** service-fed; fallback when absent.
5. Pacing/timestamps hardening; sleep/wake; endurance.

> Next doc: `driver-build-and-signing.md`. No skeleton code until WDK is in and you
> say go.

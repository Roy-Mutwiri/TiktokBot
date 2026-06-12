# RoyCam Camera Output — Phase 1 (WDK-independent)

The legitimate, own-branded software-camera frame path. The app produces frames;
this pipeline converts them and publishes them to a shared-memory ring that a
service (and, later, the **RoyCam HD Camera** AVStream driver) consumes.

This is a **software camera**. It does not impersonate any vendor, spoof any
hardware ID/signature, or claim to be a physical USB webcam. See
`docs/security-and-safety.md`.

## Data flow

```
 FrameSource ──► frame_converter ──► RingWriter ─┐
 (test/image/                                    │  shared memory
  video/app)                                     ▼  C:\Users\Public\RoyCamCamera\roycam_ring.bin
                                          [ control 64B | slot0 | slot1 | slot2 ]
                                                 │  (triple buffer + per-slot seqlock)
 RingReader ◄──── service/roycam_service.py ◄────┘
      │
      ▼  validate against the contract (shared/roycam_frame_format.h)
 DriverBridge.submit()   ← Phase-1 STUB (preview/log); becomes DeviceIoControl
                            ROYC_IOCTL_SUBMIT_FRAME when the driver is built
```

## Contract

`shared/roycam_frame_format.h` is the single source of truth; `roycam_format.py`
mirrors it byte-for-byte (control 64B, per-slot header 128B, triple buffer).
Formats: **NV12** (primary), **YUY2** (secondary), **RGB32/BGRA** (debug).

## Run it

```powershell
# 1. produce frames (test pattern proves motion end-to-end)
python -m camera_output --source test --fmt nv12 --width 1280 --height 720
#    or:  --source image --path pic.jpg     --source video --path clip.mp4

# 2. in another shell, run the frame bridge (validates + previews)
python -m service.roycam_service --preview
```

Embed in the app instead of the CLI:

```python
from camera_output import CameraOutputController, CallableSource
ctl = CameraOutputController(source=CallableSource(render_bgr_frame),
                             width=1280, height=720)
ctl.start()   # daemon thread; ctl.stop() on shutdown
```

## Dev install / uninstall

```powershell
installer\install-dev.ps1      # ring dir + per-user scheduled task (no admin, no driver)
installer\uninstall-dev.ps1    # fully reverses it
```

No kernel driver and no test-signing are involved in Phase 1.

## Tests

```powershell
python tests\test_camera_output.py     # contract sizes, format round-trips, 30fps tear-free ring
```

## What is NOT here yet

The AVStream kernel driver skeleton (`driver/`) is intentionally **not** built —
it is gated on explicit go-ahead plus dev test-signing. `DriverBridge.submit()`
is the seam where it plugs in. See `docs/driver-design.md` and
`docs/driver-build-and-signing.md`.

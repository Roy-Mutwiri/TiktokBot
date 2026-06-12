# Architecture — Camera Pipeline

How our app's generated frames become a selectable Windows camera, end to end,
with a clear kernel/user boundary and a service that keeps the camera alive
independent of the UI.

---

## 1. Layered view

```
┌──────────────────────────────────────────────────────────────────────┐
│ APP  (Python — existing)                                              │
│   avatar_studio.py / realtime_avatar.py / ai_chart / chart_pilot      │
│   IFrameSource: StaticImage | VideoFile | Webcam | Avatar | Scene |   │
│                 TradingOverlay | (Browser/Screen, if already legal)   │
│   -> resize/letterbox -> colour convert (NV12/YUY2) -> timestamp      │
│   -> write to shared-memory ring  (src/camera-output)                 │
└───────────────┬──────────────────────────────────────────────────────┘
                │  app -> service : shared-memory TRIPLE-BUFFER ring (MMF)
                │  control/status : named pipe (small messages only)
                ▼
┌──────────────────────────────────────────────────────────────────────┐
│ SERVICE  (RoyCam Frame Bridge — Windows service, native)             │
│   - owns the stable IPC endpoint (survives app restart)              │
│   - validates each frame (dims, stride, format, size, overflow-safe)  │
│   - maintains the final latest validated frame + fallback frame       │
│   - pushes latest frame to driver (IOCTL) at/around 30fps             │
│   - health/status to app; logs (no frame content)                     │
└───────────────┬──────────────────────────────────────────────────────┘
                │  service -> driver : IOCTL frame submit + shared section
                │  (control: start/stop, set-format, query-status)
                ▼  =========== KERNEL / USER BOUNDARY ===========
┌──────────────────────────────────────────────────────────────────────┐
│ DRIVER  (RoyCam AVStream minidriver — kernel, track B)               │
│   - exposes "RoyCam HD Camera" capture device (our friendly name)    │
│   - capture pin: 640x480 YUY2 / 1280x720 NV12 / 1920x1080 NV12 @30   │
│   - owns OUTPUT TIMING (paces 30fps); latest-frame-wins               │
│   - copies latest validated frame into the stream sample             │
│   - fallback internal pattern if no service/frame                     │
│   - validates ALL IOCTL input; preallocated nonpaged buffers         │
└───────────────┬──────────────────────────────────────────────────────┘
                ▼
┌──────────────────────────────────────────────────────────────────────┐
│ WINDOWS CAMERA STACK → Camera app, Chrome/WebRTC, Zoom/Teams, TikTok  │
└──────────────────────────────────────────────────────────────────────┘
```

> **Shipping baseline (track A, already built):** the same App → shared-memory →
> *user-mode* `MFCreateVirtualCamera` source (`native_camera/`) or DirectShow filter
> (`native_camera_dshow/`) — identical producer side, no kernel, no driver signing.
> Track B (the AVStream driver) is the professional upgrade; the service feeds either
> surface from the same ring.

## 2. Component responsibilities

### App (Python, existing — thin additions)
- Owns **source selection** and **frame generation** (already does).
- Adds a `camera_output_controller`: resize/crop/letterbox to the selected camera
  resolution, **colour-convert to NV12/YUY2 in user mode** (cv2/libyuv), stamp a QPC
  timestamp + monotonic frame index, write to the ring.
- **Never** talks to the driver directly. Only to the ring + the service status pipe.
- Camera output is **explicitly user-initiated** with a visible "ON AIR" indicator.

### Service (RoyCam Frame Bridge — Windows service)
- The **stable owner** of the camera feed. Survives the UI app closing/crashing.
- Validates frames; rejects unsupported sizes/formats/strides safely.
- Holds the **final validated latest frame** and the **branded fallback frame**.
- Bridges to the driver via **IOCTL** (control + frame submit). Owns service-side
  logging, health, and (admin-gated) driver start/stop helpers.
- **Security posture:** runs as a dedicated low-privilege service account where
  possible; never loads DLLs from writable dirs; never executes user scripts;
  validates everything from the app as untrusted.

### Driver (RoyCam AVStream minidriver — kernel)
- The **camera device**. Owns **output timing** (paces the selected fps) so playback
  is smooth even if the app stutters.
- On each capture-pin sample request: copy the latest validated frame into the
  sample (no per-frame allocation, no blocking, preallocated nonpaged buffers,
  overflow-safe size math). If stale/absent → **fallback frame**.
- **Trusts nothing from user mode.** Every IOCTL: verify method, validate
  `InputBufferLength`/`OutputBufferLength`, validate width/height/stride/format
  against the negotiated media type, bound the copy.

## 3. Why a service (not app-direct-to-driver)
- The camera must not die when the UI process exits or a client opens the camera
  before the app starts. A service is the durable owner.
- Centralizes validation, logging, fallback, and the single driver control channel.
- Lets multiple UI sessions / a future headless mode feed the same camera.
- Mirrors what `vcam_host.exe` hints at today, but as a proper, restartable service.

## 4. Format & conversion strategy
- **Convert in user mode** (app or service) using cv2/libyuv. Kernel stays simple
  and only ever sees already-formatted NV12/YUY2 (or RGB32 for debug).
- Camera advertises, in priority order: **NV12 (primary)**, **YUY2 (secondary)**,
  **RGB32 (debug only)**. MJPEG optional later.
- Resolutions M3 target: 640×480 YUY2, 1280×720 NV12, 1920×1080 NV12, all @ 30fps
  (15fps selectable).

## 5. Frame pacing & timestamps
- **Driver paces output** at the negotiated fps (e.g., 33.33 ms). It does *not* depend
  on app frame timing.
- If the latest frame is older than a timeout (e.g., > 2 frame intervals), the driver
  **repeats the last frame**, and after a longer timeout serves the **fallback frame**.
- Timestamps: **monotonic** (QPC-derived), consistent frame duration, never negative
  or duplicate where clients dislike it. The app's QPC stamp is informational; the
  driver assigns the authoritative presentation time.

## 6. Threading & ownership (no blocking in hot paths)
- **App:** producer thread writes one ring slot per frame; never blocks on the service.
- **Service:** one receiver loop (drains ring, validates, updates final frame); one
  driver-push loop (IOCTL at ≤ fps). Status pipe on its own thread.
- **Driver:** streaming callback only copies preallocated memory; frame updates land
  via IOCTL on a separate path guarded by a lightweight lock; the streaming callback
  takes the lock for a bounded copy only — **no allocation, no I/O, no long holds.**

## 7. Failure isolation (summary; full in `uninstall-and-recovery.md`)
- App closes/crashes → ring goes stale → service serves fallback → driver shows
  fallback frame. **No black/frozen confusion, no crash.**
- Service closes/crashes → driver detects stale submit → fallback pattern; service
  auto-restarts (SCM recovery).
- Driver must **never** bugcheck on bad user-mode input — validate and reject.

## 8. Privacy & transparency
- Output is opt-in, with a clear active indicator and a Stop control.
- No hidden capture; logs record events/metrics, **not** frame pixels or user content.
- Camera name is honestly ours ("RoyCam HD Camera"); docs call it a software camera.

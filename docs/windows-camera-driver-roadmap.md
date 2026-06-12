# Windows Custom Camera Driver — Roadmap

**Product camera name (configurable):** `RoyCam HD Camera` (alt: `TradeFix HD Camera`).
**Branding rule:** this is **our own software-backed camera device**. We never
impersonate another vendor, never spoof VID/PID/hardware IDs/signatures, and the
docs always describe it honestly as a software camera driver.

This roadmap supersedes nothing already shipped — it adds a **kernel-mode AVStream
camera** as the "professional driver" track while keeping the existing user-mode
software cameras as the safe, shippable baseline.

---

## 0. Honest framing (read first)

Two viable "camera surfaces" exist for software frames on Windows:

- **(A) User-mode software camera — already built.** `MFCreateVirtualCamera` media
  source (`native_camera/`) and/or DirectShow source filter (`native_camera_dshow/`).
  Works on normal machines, **no kernel, no special driver signing**. Microsoft's
  sanctioned route for *software* frame sources. This is the recommended **shipping**
  surface.
- **(B) Kernel-mode AVStream minidriver — this roadmap's new work.** A real camera
  device in Device Manager, maximally hardware-like. **Requires the WDK, kernel C++,
  and Microsoft driver signing to run on end-user machines.** This is the
  professional-but-heavy track.

We pursue (B) deliberately, but every phase keeps (A) as the working fallback so the
product is never blocked on driver signing.

**The single biggest risk is driver signing/distribution, not the code.** See
`driver-build-and-signing.md`. Do not let anyone believe an AVStream driver can be
shipped to users without Microsoft-signed packages.

---

## Phase status at a glance

| Phase | Title | Status |
|------:|-------|--------|
| 0 | Repository discovery | **Done** → `repo-analysis.md` |
| 1 | User-mode frame pipeline (30fps, formats, ring, fallback) | **~80% exists** (`avatar_sharedframe.py`, BGR producers) — needs format converter + ring + telemetry |
| 2 | IPC / shared-memory design | **Single-frame exists** — needs v2 header + triple buffer → `ipc-frame-transport.md` |
| 3 | Windows service bridge | **Not started** (logic is today inline in `vcam_host`/app) |
| 4 | AVStream driver skeleton (simulated camera, our name) | **Not started; WDK not installed** |
| 5 | Driver frame-source integration | Not started |
| 6 | App UI (Camera Output tab) | Not started (Studio UI exists to extend) |
| 7 | Installer + admin flow | **Dev scripts exist** for MF/DShow; service+driver installer not started |
| 8 | Driver signing plan | Doc pending → `driver-build-and-signing.md` |
| 9 | Testing matrix | Doc pending → `testing-matrix.md` |
| 10 | Stability & recovery | Partial (fallback frame exists in DShow filter) |

---

## Proposed folder structure

```
TiktokBot/
├─ docs/                          # design docs
│   ├─ repo-analysis.md
│   ├─ windows-camera-driver-roadmap.md   (this file)
│   ├─ architecture-camera-pipeline.md
│   ├─ ipc-frame-transport.md
│   ├─ driver-design.md
│   ├─ driver-build-and-signing.md
│   ├─ testing-matrix.md
│   ├─ security-and-safety.md
│   └─ uninstall-and-recovery.md
│
├─ shared/                        # language-neutral contracts (C/C++ headers)
│   ├─ roycam_frame_format.h      # frame header + pixel-format enums
│   ├─ roycam_ioctl.h             # IOCTL codes + control structs
│   └─ roycam_status_codes.h      # status/error enums shared app↔service↔driver
│
├─ service/                       # RoyCam Frame Bridge (Windows service)
│   ├─ RoyCamService.sln
│   ├─ frame_receiver.*           # owns the app↔service shared-memory ring
│   ├─ driver_bridge.*            # service↔driver: IOCTL + shared section
│   ├─ service_status.*           # health endpoint (named pipe, status only)
│   ├─ service_config.*           # camera name / resolution / fps config
│   └─ fallback_frame.*           # branded "waiting for source" frame
│
├─ driver/
│   └─ RoyCamDriver/              # SEPARATE WDK/AVStream solution
│       ├─ RoyCamDriver.sln
│       ├─ RoyCamDriver.vcxproj
│       ├─ RoyCam.inx / RoyCam.inf
│       ├─ device.*  capture.*  frame_buffer.*  ioctl.*  media_formats.*  trace.*
│       └─ public/ (roycam_ioctl.h, roycam_public.h)   # mirrors /shared
│
├─ installer/
│   ├─ dev-install.ps1   dev-uninstall.ps1
│   ├─ install-service.ps1  uninstall-service.ps1
│   └─ installer-notes.md
│
├─ tests/
│   ├─ frame-pipeline-tests/  shared-memory-tests/  service-tests/  endurance-tests/
│
├─ src/camera-output/            # Python side (thin)
│   ├─ camera_output_controller.py   # start/stop, status, source selection
│   ├─ frame_pipeline.py             # resize/letterbox + colour convert (libyuv/cv2)
│   └─ frame_shared_memory_client.py # wraps avatar_sharedframe (v2 header)
│
└─ (existing Python app, engines/, native_camera*/ stay as-is)
```

`/shared` is the single source of truth; `driver/RoyCamDriver/public/` and the
Python client are kept byte-compatible with it (same struct layout, same enums).

---

## Milestone checklist

**M0 — Docs & decision (this deliverable)**
- [x] `repo-analysis.md`
- [x] `windows-camera-driver-roadmap.md`
- [x] `architecture-camera-pipeline.md`
- [x] `ipc-frame-transport.md`
- [x] `security-and-safety.md`
- [ ] Stakeholder sign-off on (A)-baseline + (B)-AVStream dual track *(awaiting your go)*

**M1 — User-mode pipeline hardened**
- [ ] `roycam_frame_format.h` v2 (magic `ROYC`, full header)
- [ ] Triple-buffer ring (atomic write index, latest-wins) replacing single-frame seqlock
- [ ] Frame converter BGR→NV12/YUY2/RGB32 (user mode, cv2/libyuv) + unit tests
- [ ] Color-bar / timestamp / counter test source
- [ ] Static-image + video-file sources
- [ ] 10-minute 30fps stress test green

**M2 — Service bridge**
- [ ] `RoyCamService` skeleton (install/start/stop/uninstall dev scripts)
- [ ] Frame receiver (app→service ring), validation, fallback frame
- [ ] Health/status over named pipe (status only, no frame data)
- [ ] Survives app close (serves fallback)

**M3 — AVStream driver (simulated)** *(blocked on WDK install + test-signing)*
- [ ] WDK installed; driver builds
- [ ] `RoyCam HD Camera` appears in Device Manager
- [ ] Windows Camera app opens it; outputs internal test pattern
- [ ] Media types: 640×480 YUY2, 1280×720 NV12, 1920×1080 NV12 @ 30fps
- [ ] Clean start/stop, stable timestamps, frame pacing

**M4 — Driver fed by service**
- [ ] IOCTL frame-submit (service → driver), validated, preallocated buffers
- [ ] Latest-frame copy into stream sample in streaming callback (no alloc, no block)
- [ ] Fallback test pattern when no service/source
- [ ] 30fps stability test green

**M5 — App integration**
- [ ] Camera Output tab (start/stop, source, resolution, fps, status, fallback image)
- [ ] `IFrameSource` abstraction + concrete sources
- [ ] Status surface (driver installed / service running / fps / last-frame / errors)

**M6 — Installer + signing + test matrix**
- [ ] Dev install/uninstall (pnputil + sc) clean
- [ ] `driver-build-and-signing.md` production checklist
- [ ] `testing-matrix.md` executed (Camera app → Chrome/WebRTC → Zoom/Teams → endurance)
- [ ] `uninstall-and-recovery.md` verified (no broken devices)

---

## "What I need to change in the repo" — summary

Nothing destructive; everything is **additive**. Concretely:

1. **Add folders** `/shared`, `/service`, `/driver`, `/installer`, `/tests`,
   `src/camera-output/` (above). No existing file is deleted.
2. **Evolve the frame transport (additive, versioned).** Introduce
   `shared/roycam_frame_format.h` (v2, magic `ROYC`, richer header + triple buffer).
   Add a Python writer that speaks v2 alongside the current `avatar_sharedframe.py`
   v1 (`AVC1`) so existing MF/DShow cameras keep working during migration.
3. **Add a producer hook in the app** (no behavior change unless enabled): the
   render loops already compute `final` BGR frames; route them to
   `camera_output_controller` when the (new, opt-in) "Camera Output" toggle is on —
   defaulting OFF, exactly like the existing `AVATAR_AICHART`/`AVATAR_TV` env gates.
4. **Add a `.gitignore` entry** for build artifacts (`/driver/**/x64/`, `*.sys`,
   `*.cat`, `/service/**/bin/`) so kernel/service builds don't get auto-committed
   (the repo already had a large-file incident; keep binaries out — see memory note
   on `git-large-file-push-block`).
5. **Install the WDK** on the dev machine (prerequisite for any kernel work). This is
   an environment change, not a repo change, but it gates Phase 4+.
6. **Do not** add admin-at-startup, driver auto-load tricks, or anything that runs
   without explicit user action. Camera output is user-initiated and visibly
   indicated (see `security-and-safety.md`).

---

## Technique comparison (your 1–30 list) — decisions

| # | Technique | Decision |
|---|-----------|----------|
| 1 | AVStream camera minidriver | **Primary (track B).** Most professional; gated on WDK + signing. |
| 2 | AvsCamera sample as base | **Yes**, conceptual reference for the skeleton (simulated pin → our name → static → fed). |
| 3 | MF virtual camera API | **Already built; keep as shipping baseline (track A) / fallback.** Not "primary" per your note, but it is what ships today. |
| 4 | DirectShow virtual filter | **Already built (`native_camera_dshow`).** Keep for legacy/DShow apps. Not primary. |
| 5 | UVC hardware device | **Out of scope** (needs hardware). Documented only. |
| 6 | USB device emulation | **Not implemented.** Complex/fragile; only after separate explicit approval. |
| 7 | Shared-memory ring buffer | **Yes — primary app↔service transport.** |
| 8 | Named pipes | **Control/status only**, not bulk frames. |
| 9 | IOCTL frame submit | **Yes — primary service↔driver path** (validated boundary). |
| 10 | Memory-mapped file | **Yes — app↔service bridge** (evolution of current `frame.bin`). |
| 11 | Windows service bridge | **Strongly yes.** Decouples camera from UI process; the big stability win. |
| 12 | User-mode frame conversion | **Yes.** Keep kernel simple; convert in service/app (cv2/libyuv). |
| 13 | Kernel-mode conversion | **Avoid.** Driver prefers already-NV12/YUY2 frames. |
| 14 | NV12 | **Primary** camera format. |
| 15 | YUY2 | **Secondary** (broad webcam compat). |
| 16 | RGB32 | **Debug/fallback only.** |
| 17 | MJPEG | **Optional later** (encode cost). |
| 18 | Fixed frame pacing in driver | **Yes — driver owns 30fps timing**, independent of app jitter. |
| 19 | Latest-frame-wins | **Yes.** |
| 20 | Triple buffering | **Yes** (app↔service). |
| 21 | Fallback frame | **Required** — branded "RoyCam HD Camera — waiting for source". |
| 22 | Health monitoring | **Yes** across app↔service↔driver. |
| 23 | Install/uninstall scripts | **Yes** (dev pnputil/sc; prod installer later). |
| 24 | Test signing | **Dev only** (watermark; dev machine). |
| 25 | Production MS signing | **Required for distribution** of the AVStream driver. |
| 26 | Windows Camera app test | **Yes** (first client). |
| 27 | Chrome/WebRTC test | **Yes.** |
| 28 | TikTok Studio test | **Yes** (target app). |
| 29 | Endurance test | **Yes** (1h → 6h). |
| 30 | Safe rollback | **Yes** — uninstall must leave no broken devices. |

---

## Next deliverables (after your go-ahead)

In order, and **not before you say so**: `driver-design.md`, then the
**Phase 1 user-mode pipeline + v2 transport header**, then the **service skeleton**,
then — only after the **WDK is installed and test-signing is enabled** — the
**AVStream driver skeleton** (simulated camera, our name).

# Phase 0 — Repository Analysis

**Date:** 2026-06-12
**Scope:** Determine whether this repo is suitable for a custom Windows camera
driver pipeline, and where the new native components should live. No assumptions —
this is based on an actual inspection of the tree.

---

## 1. What this app is

| Aspect | Finding |
|--------|---------|
| **Type** | Windows desktop application — an AI avatar / "Avatar Studio" for live streaming + a trading-analysis overlay. |
| **Language / runtime** | **Python 3.11 (CPython)**, system install at `C:\Users\user\AppData\Local\Programs\Python\Python311`. |
| **UI framework** | **Tkinter** (`avatar_studio.py`, custom Canvas-drawn HUD). Headless pipeline in `realtime_avatar.py`. |
| **Other languages** | **C++** for the existing native camera bits (`native_camera/`, `native_camera_dshow/`). **No C#, Electron, Node, or web stack.** |
| **OS support** | **Windows-only** (uses `winsound`, `win32` paths, `pyvirtualcam`, a DirectShow filter, registry registration). |
| **Build system** | Python: none (run with `python avatar_studio.py`). Native: PowerShell + MSBuild/`cl.exe` (`native_camera*/build.ps1`). No CMake, no `.sln` at root. |
| **Packaging / installer** | **None today.** No MSI / Inno / MSIX. Distribution = clone repo + run. `autosync.py` auto-commits/pushes to GitHub. |
| **Admin usage** | Already present but scoped: `native_camera*/install.ps1` and `brand_camera.py` **self-elevate (UAC `RunAs`)** to write registry keys for camera registration. No always-on admin. |

## 2. Does the app already produce video frames? — **Yes, extensively**

The whole product is a frame generator. The pipeline computes **BGR `numpy` frames**
(default 512×512, `FRAME_SIZE`) every tick:

- `realtime_avatar.run()` — webcam → LivePortrait → MuseTalk mouth → enhance → `final` BGR frame.
- `avatar_studio.py._loop()` — the GUI variant.
- `engines/ai_chart.py`, `chart_pilot.py` — rendered trading-chart scenes (BGR frames).
- Sources already exist conceptually: webcam, AI avatar, rendered scene, trading overlay, (browser/screen via the TradingView pilot).

**Existing media libraries:** OpenCV (`cv2`), `numpy`, `pyvirtualcam`, `sounddevice`,
`soundfile`, `ffmpeg` (external binary), `torch+cu128`, Playwright (browser).

## 3. Existing camera-output surface — **this is the important part**

The repo has **already built three software-camera surfaces**. We are not starting
from zero; most of Phases 1–3 of the roadmap exist in user mode:

1. **`pyvirtualcam` → "OBS Virtual Camera"** (DirectShow, requires OBS installed) — the original output. *This is the OBS dependency the new plan wants to remove.*
2. **`native_camera/`** — a **Media Foundation `MFCreateVirtualCamera` software media source** (C++ DLL `VirtualCameraMediaSource.dll` + `vcam_host.cpp` session-lifetime host). It reads frames from a shared file and is a **working, branded, system-detected camera** ("Avatar Studio Camera (Windows Virtual Camera)").
3. **`native_camera_dshow/`** — a **DirectShow source filter** (`CSource`/`CSourceStream`, `AvatarCamFilter.cpp`) named **"Avatar Studio Camera"**, no virtual-camera flag, registered via `DllRegisterServer`. Also working.

**Frame transport already implemented:** `avatar_sharedframe.py` (writer) + the C++
`SharedFrame.h` (reader) — a **file-backed memory-mapped shared buffer** at
`C:\Users\Public\AvatarStudioCamera\frame.bin`:

- Header (64 bytes): magic `'AVC1'` (`0x31435641`), version, width, height, fourcc (0 = RGB32/BGRA top-down).
- **Seqlock** at offset 24 (odd = write in progress, even = stable) → tear-free single-frame latest-wins.
- Payload: `width*height*4` BGRA bytes.

This is **Phase 1 (frame generation) and Phase 2 (IPC) already done** in a single-frame
form. The natural producer hook is `SharedFrameWriter.write(bgr)` /
`_SharedMemCam.send(frame)` in `avatar_camera.py`.

## 4. The gap vs. the requested AVStream driver

What's requested is the **professional kernel-mode AVStream camera minidriver**
("RoyCam HD Camera" / "TradeFix HD Camera") — a real camera *device* in Device
Manager, not the user-mode software cameras above. That path is **not started** and
has hard prerequisites this machine does **not** currently meet:

| Prerequisite | Status |
|--------------|--------|
| **Windows Driver Kit (WDK)** matching the SDK | **NOT installed.** Only the user-mode Windows SDK 10.0.26100 is present (no `…\Include\<ver>\km` kernel headers, no `stampinf`/`inf2cat` WDK build steps). The existing user-mode C++ camera bits build with VS Build Tools + SDK; a kernel driver **cannot** build until the WDK is installed. |
| **Test signing for dev** (`bcdedit /set testsigning on`) | Not configured; required to load a self-signed driver during development (puts a desktop watermark; dev machine only). |
| **Production driver signing** | Not available without an **EV code-signing cert + Microsoft Partner Center (Hardware Dev Center)** attestation/WHQL signing. **This is the real distribution wall** (see `driver-build-and-signing.md`). |
| **VS Build Tools 2022 + Win11 SDK** | **Present** (user-mode C++ builds work today). |

## 5. Suitability verdict

- **The Python repo is the correct "app" layer** and stays as-is. It already
  generates frames and already knows how to publish them to a shared buffer.
- **Driver work must be a separate native (C++/WDK) Visual Studio solution.** Do
  **not** intermix kernel C++ into the Python tree. Co-locate it in this repo under
  `/driver`, `/service`, `/shared`, `/installer` (proposed below) but keep it as its
  own `.sln`.
- **Strong recommendation:** keep the **existing user-mode software camera
  (MF + DShow) as the shipping product** — it is branded, works on normal machines
  with **no driver-signing wall**, and already removes the OBS dependency (option 2
  above is OBS-free). Pursue the **AVStream kernel driver as a parallel R&D track**
  with the signing/WDK reality understood up front. If "a camera device that shows
  in Device Manager / behaves maximally like hardware" is a hard requirement, AVStream
  is the route — but it cannot be distributed to end users without Microsoft-signed
  driver packages.

## 6. Reusable assets already in the repo

| Asset | Reuse as |
|-------|----------|
| `avatar_sharedframe.py` + `native_camera/SharedFrame.h` | Basis for the documented v2 frame transport (`roycam_frame_format.h`), evolved single-frame → triple-buffer ring. |
| `native_camera_dshow/AvatarCamFilter.cpp` | Reference for capture-pin media types (YUY2/NV12/RGB), `IAMStreamConfig`, `PIN_CATEGORY_CAPTURE` — directly informs the driver's media-format table. |
| `native_camera/SimpleFrameGenerator.cpp`, `vcam_host.cpp` | Reference for the MF software-camera fallback path and the shared-file reader pattern. |
| `native_camera*/install.ps1` | Pattern for self-elevating dev install/uninstall scripts (`pnputil`, `sc`). |
| `_SharedMemCam` / `SharedFrameWriter` | The producer hook the new service/driver consumes. |

## 7. Recommended top-level layout to add

```
/docs            design docs (this folder)
/service         RoyCam Frame Bridge — Windows service (native C++ or .NET)
/driver          RoyCam AVStream minidriver (separate VS/WDK solution)
/shared          roycam_frame_format.h, roycam_ioctl.h, roycam_status_codes.h
/installer       dev install/uninstall (pnputil + sc), production installer notes
/tests           frame-pipeline / shared-memory / endurance tests
src/camera-output  thin Python module that wraps avatar_sharedframe + status
```

The Python app keeps its current root layout; only `src/camera-output` (or a flat
`camera_output/` module) is added on the Python side as the controller/wrapper.

> Next docs: `windows-camera-driver-roadmap.md`, `architecture-camera-pipeline.md`,
> `ipc-frame-transport.md`, `security-and-safety.md`.

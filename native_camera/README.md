# Avatar Studio Camera — native virtual camera (appears only while running)

This builds a **real Windows 11 virtual camera** using `MFCreateVirtualCamera`.
Unlike the OBS-based bridge, this device:

- **Appears in every app's camera list the moment you start the avatar**
  (Zoom, Teams, Google Meet, Windows Camera, Chrome/Edge, Discord, OBS, …), and
- **Disappears when you stop it** (lifetime = *Session*), and
- Carries **our own name** — "Avatar Studio Camera" — not OBS's.

It works because the camera is backed by a tiny native **media source DLL**
(adapted from Microsoft's official VirtualCamera sample) that runs inside the
Windows Camera Frame Server and reads each live frame out of a shared memory
file the Python avatar writes.

```
 avatar_camera.py --native ─► avatar_sharedframe.py ─► C:\Users\Public\AvatarStudioCamera\frame.bin
                                                                  ▲
 vcam_host.exe ─► MFCreateVirtualCamera(Session) ─► Frame Server loads
                                                    VirtualCameraMediaSource.dll
                                                    (SharedFrame.h reads the file)
```

## Why a build is required

A camera device must be backed by **registered native code** — there is no
pure-Python way to make one appear. That native piece needs a C++ toolchain,
which isn't installed yet. This is a **one-time setup**.

## 1. Install the toolchain (one time, needs admin/UAC)

```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools -e `
  --accept-source-agreements --accept-package-agreements `
  --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.Windows11SDK.22621 --includeRecommended"
```

(≈3–7 GB. The Windows 11 SDK provides `mfvirtualcamera.h` / `MFCreateVirtualCamera`.)

## 2. Build

```powershell
powershell -ExecutionPolicy Bypass -File native_camera\setup.ps1   # clone sample + graft our files
powershell -ExecutionPolicy Bypass -File native_camera\build.ps1   # -> VirtualCameraMediaSource.dll + vcam_host.exe
```

## 3. Register the media source (one time, admin)

```powershell
powershell -ExecutionPolicy Bypass -File native_camera\install.ps1
```

## 4. Go live

```powershell
python avatar_camera.py --native
```

"Avatar Studio Camera" now appears while this runs. Pick it in any app; you see
the live avatar. Stop the script → the camera is gone. To unregister the source
entirely: `install.ps1 -Uninstall`.

## Files

| file | role |
|------|------|
| `SharedFrame.h` | native reader of the shared frame file (mirrors `avatar_sharedframe.py`) |
| `SimpleFrameGenerator.cpp` | adapted sample file — copies the avatar frame instead of a test pattern |
| `vcam_host.cpp` | creates the session-lifetime virtual camera and holds it open |
| `setup.ps1` / `build.ps1` / `install.ps1` | fetch + graft, compile, register |

## Status: BUILT & VERIFIED on this machine (2026-06-10)

The full path is working and tested end-to-end:
- VS Build Tools 2022 + Win11 SDK 10.0.26100 installed; DLL + `vcam_host.exe`
  compiled clean.
- The DLL is installed to `C:\Program Files\AvatarStudioCamera\` and registered
  (InprocServer32). It MUST live in a system path - the Frame Server service
  cannot load a media source from a user-profile folder (that caused an early
  `BindToObject` / `E_ACCESSDENIED` failure).
- Verified with ffmpeg: the camera appears as "Avatar Studio Camera (Windows
  Virtual Camera)" at 512x512 @ 30fps, streams the exact fed frames (BLUE and
  GREEN color round-trips confirmed across repeated trials), and vanishes the
  instant `vcam_host.exe` exits (Session lifetime).

Notes if you rebuild against a newer SDK:
- `setup.ps1` patches the resolution (NUM_IMAGE_ROWS/COLS -> 512) and removes the
  sample's duplicate `MF_VIRTUALCAMERA_*` GUID definitions (the SDK provides them).
- `build.ps1` retargets WindowsTargetPlatformVersion to the installed SDK and
  restores packages.config NuGet packages (CppWinRT, wil) via a fetched nuget.exe.
- `IMFVirtualCamera::Start()` is required for the camera to surface (it may return
  E_ACCESSDENIED but still works - the host logs and continues).

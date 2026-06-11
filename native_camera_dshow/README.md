# Avatar Studio Camera - DirectShow (detected as a normal webcam)

This builds a **DirectShow software camera** named **"Avatar Studio Camera"**.
Unlike the `MFCreateVirtualCamera` device (`native_camera/`), this one:

- has a **clean name** - no "(Windows Virtual Camera)" suffix, and
- carries **no virtual-camera flag**, so apps that flag/block virtual cameras
  see it as an ordinary webcam.

It works in every app that uses DirectShow: **OBS, Zoom, Discord, Chrome/Edge,
TikTok Live Studio, Teams (classic)**. It does NOT appear in Frame-Server-only
apps (the built-in Windows Camera app, newest Teams) - for those, use the MF
camera in `native_camera/` (which those apps accept, just labelled virtual).

Frames come from the same shared file the avatar writes (`avatar_sharedframe.py`),
so a single `python avatar_camera.py --dshow` feeds it.

```
 avatar_camera.py --dshow ─► C:\Users\Public\AvatarStudioCamera\frame.bin
                                          ▲
 your app opens "Avatar Studio Camera" ─► AvatarCamFilter.dll (DirectShow source,
                                          reads the shared file) ─► RGB24 512x512@30
```

## Build & install (one time)

Requires VS Build Tools + Windows SDK (already installed for the MF camera).

```powershell
powershell -ExecutionPolicy Bypass -File native_camera_dshow\build.ps1     # -> AvatarCamFilter.dll
powershell -ExecutionPolicy Bypass -File native_camera_dshow\install.ps1   # ADMIN: register it
```

## Use

```powershell
python avatar_camera.py --dshow
```

Then pick **Avatar Studio Camera** in your app. Remove it with
`install.ps1 -Uninstall`.

## How it's built

- `AvatarCamFilter.cpp` - a `CSource`/`CSourceStream` filter. The output pin
  implements `IKsPropertySet` (reports `PIN_CATEGORY_CAPTURE`) and
  `IAMStreamConfig` so apps treat it as a real capture device. `FillBuffer`
  copies the latest avatar frame (BGRA top-down) into RGB24 bottom-up.
- `baseclasses/` - Microsoft's DirectShow base classes (compiled to strmbase.lib).
- `build.ps1` compiles base classes once (cached), then the filter DLL (static
  CRT so it loads cleanly into any host app).
- `install.ps1` calls the DLL's own `DllRegisterServer` directly (more reliable
  than regsvr32) to add it to the video-input-device category.

## Status: BUILT & VERIFIED (2026-06-11)

Verified by building a real DirectShow capture graph (source -> Sample Grabber ->
null renderer): connects, runs, and the grabbed buffer is the exact fed frame
(786432 bytes RGB24, color round-trip exact). Enumerates as a clean
"Avatar Studio Camera" with no virtual suffix.

Note: ffmpeg 8.1's `dshow` input rejects the pin ("could not find output pin") -
that is an ffmpeg-specific heuristic; real DirectShow graphs (OBS/Zoom/etc.)
connect and stream fine, as proven by the Sample Grabber test.

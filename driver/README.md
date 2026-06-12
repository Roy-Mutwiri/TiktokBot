# RoyCam HD Camera — AVStream minidriver (skeleton)

A **legitimate, own-branded software camera**. Root-enumerated under our own
hardware ID (`root\RoyCamHDCamera`), advertises `KSCATEGORY_VIDEO_CAMERA` +
`KSCATEGORY_CAPTURE` + `KSCATEGORY_VIDEO`, and exposes app-generated frames on a
capture pin. It does **not** impersonate any vendor, spoof any VID/PID/hardware
ID/signature, and does **not** claim to be physical USB hardware.

## Frame source stages

| Stage | Provider | Status |
|-------|----------|--------|
| 1 | `RoycFillSimulatedFrame` — driver-generated color bars + moving marker | **active** (ships in the skeleton) |
| 2 | static embedded image | wired, inert |
| 3 | service-fed via IOCTL (`ROYC_IOCTL_SUBMIT_FRAME`) from the RoyCam Frame Bridge | wired (`ioctl.c` + `framesource.c`), not yet routed into an IRP dispatch |

Stage 3 is the seam where `service/roycam_service.py`'s `DriverBridge.submit()`
will `DeviceIoControl` the driver. The contract is `shared/roycam_ioctl.h`.

## Files

| File | Role |
|------|------|
| `driver.c` | `DriverEntry` → `KsInitializeDriver` |
| `device.c` | KS device dispatch + device descriptor |
| `filter.c` | filter + capture-pin descriptors, categories, allocator framing |
| `capture.c` | pin lifecycle, 30fps timer pacing, frame `Process`, format negotiation, intersect handler |
| `formats.c` | `KS_DATARANGE_VIDEO` tables (NV12/YUY2/RGB32) + size/stride helpers |
| `framesource.c` | simulated pattern generator + validated service-fed copy |
| `ioctl.c` | bounds-checked service IOCTL handling (stage 3) |
| `roycam.h` / `trace.h` | shared defs + `DbgPrintEx` tracing |
| `RoyCamDriver.inf` | install: Camera class, own branding, KS interfaces |

## Build prerequisites — IMPORTANT

The WDK **headers/libs/build-props** are installed
(`winget install Microsoft.WindowsWDK.10.0.26100`), and `ks.h`/`ksmedia.h`/`km`
all resolve from `10.0.26100.0`. **But** building a kernel `Driver` project with
msbuild also needs the `WindowsKernelModeDriver10.0` **platform toolset**, which
ships in the **WDK Visual Studio extension** (a `.vsix`). On a *Build Tools-only*
box that extension is not present, so msbuild fails with:

```
error MSB8020: The build tools for WindowsKernelModeDriver10.0 ... cannot be found
```

To unblock the build, install the WDK VS extension into the VS/Build Tools
instance (one of):
- Install **Visual Studio 2022** (Community) with the *Desktop C++* workload,
  then the **WDK** (its installer registers the VS extension), **or**
- Install the standalone *Windows Driver Kit* VS extension `.vsix` into the
  existing Build Tools via `VSIXInstaller.exe`.

Everything else (sources, INF, project, build/sign/install scripts) is ready;
once the toolset resolves, `build.ps1` produces the `.sys` with no code changes.

## Build → sign (dev) → install

```powershell
driver\build.ps1 -Config Release                 # -> RoyCamDriver\x64\Release\RoyCamDriver\*.sys/.inf/.cat
driver\sign-testcert.ps1 -Config Release         # DEV self-signed test cert (Test Mode watermark)
# one-time, operator-performed, then REBOOT:
#   bcdedit /set testsigning on
driver\install.ps1 -Config Release               # pnputil + devgen root\RoyCamHDCamera
driver\uninstall.ps1                             # full reversal
```

**Signing reality:** self-signed test signing is dev-box only. Production
distribution requires an **EV certificate + Microsoft Partner Center
attestation** (or WHQL). See `docs/driver-build-and-signing.md`.

## Status of this skeleton

The code follows the documented AVStream/KS contract and was authored against
the `10.0.26100.0` headers (struct layouts, dispatch tables, and the framing/
intersect APIs were checked against `ks.h`/`ksmedia.h`). It has **not yet been
compiled** because the toolset above is missing — treat a first compile + SDV +
Driver Verifier pass as the next step once the extension is installed.

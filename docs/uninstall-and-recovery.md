# Uninstall & Recovery

Two goals: (1) the system **never** ends up with a broken camera device, and (2) the
camera **never** crashes Windows or shows a frozen/black feed when something upstream
fails. Covers clean removal and runtime failure handling.

---

## 1. Runtime failure handling (never crash, never freeze)

| Situation | Behavior |
|-----------|----------|
| App closes / crashes | Shared ring goes stale → service serves the **branded fallback frame** → driver outputs fallback. No black/frozen feed, no crash. |
| Service closes / crashes | Driver detects stale submit (no IOCTL within `T_fallback`) → internal fallback pattern. SCM **auto-restarts** the service (recovery actions: restart/restart/restart). |
| Source resolution changes | Validate **before** accepting; if it doesn't match the negotiated pin media type, **reject** and keep the last good format. |
| Invalid / oversized frame | **Reject safely** (logged, counted); stream continues on the last good frame. Bounded copy only. |
| Client opens before app starts | Fallback frame shown immediately (camera always "works"). |
| Client changes resolution | Pin selects a **supported** media type; never fails the open. |
| Multiple clients | v1 = single-client; 2nd open **fails gracefully** (documented), no crash. |
| Driver fails to load | App detects via status (`driver installed = false`) and **falls back to the user-mode camera** (MF/DShow) automatically. |

**Crash-safety invariants (driver):** validate all IOCTL input; overflow-safe math;
preallocated buffers (no per-frame alloc); no blocking in streaming callback; release
everything on stop/remove. (See `security-and-safety.md` §3, `driver-design.md` §5/§6.)

## 2. Status codes surfaced to the app

`/shared/roycam_status_codes.h` (shared enum), shown in the Camera Output tab:
`OK`, `DRIVER_NOT_INSTALLED`, `SERVICE_STOPPED`, `NO_SOURCE`, `FRAME_TOO_LARGE`,
`UNSUPPORTED_FORMAT`, `RESOLUTION_MISMATCH`, `IPC_TIMEOUT`, `CLIENT_BUSY`,
`FALLBACK_ACTIVE`. The app shows: driver installed?, service running?, source
connected?, current fps, last-frame age, last error.

## 3. Clean uninstall — order matters

Remove **app → service → driver** (reverse of install). Each step verifies success
and leaves nothing behind.

### Development (scripts in `/installer`)

```powershell
# 1) stop using the camera (app)            -> close app / Stop Camera Output

# 2) service: stop, then remove
sc.exe stop  RoyCamService
sc.exe delete RoyCamService

# 3) driver (track B): find the oem*.inf, uninstall + delete the package
pnputil /enum-drivers | findstr /i RoyCam            # -> oemNN.inf
pnputil /delete-driver oemNN.inf /uninstall          # removes the device too

# 4) (track A user-mode camera) unregister, e.g.
#    native_camera_dshow\install.ps1 -Uninstall   /   native_camera\install.ps1 -Uninstall

# 5) verify nothing remains
pnputil /enum-drivers | findstr /i RoyCam            # -> (none)
# Device Manager: no "RoyCam HD Camera"; no ghosted/hidden device
```

### Production (installer)
- WiX/Inno (or MSIX + a separate driver step) performs the same sequence with
  rollback. If any step fails, the installer **reverts** the prior steps so the
  machine is never left half-installed.

## 4. Recovery from a bad state

| Symptom | Fix |
|---------|-----|
| Ghosted/hidden "RoyCam HD Camera" after a failed uninstall | Device Manager → View → Show hidden devices → uninstall device **+ "delete driver software"**; then `pnputil /delete-driver oemNN.inf /force`. |
| Driver won't load (code 52, unsigned) on a dev box | Confirm test cert in Root+TrustedPublisher and `bcdedit /set testsigning on` + reboot (dev only). On a normal machine this is expected → use the Microsoft-signed package or the user-mode camera. |
| Service stuck / won't start | `sc query RoyCamService`; check service log; `sc delete` + reinstall; ensure binary path + account valid. |
| Camera frozen on fallback forever | App not producing or wrong format → check status codes (`NO_SOURCE`/`UNSUPPORTED_FORMAT`); restart app/Camera Output. |
| Whole driver track blocked (signing/instability) | **Fall back to the user-mode camera** (already shipping) — no kernel, no driver, no reboot. |

## 5. Rollback principle
- **Additive, reversible, ordered.** Installs and uninstalls are scripted and
  idempotent. A failed install rolls back to the pre-install state. A failed driver
  never blocks the product because the **user-mode camera is always available** as the
  baseline. No step requires disabling Secure Boot or leaving test-signing on for end
  users.

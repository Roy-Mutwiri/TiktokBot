# Testing Matrix

Run in **this exact order** — each step gates the next. Stop and fix on first hard
failure. Applies to both the user-mode camera (track A, today) and the AVStream
driver (track B); steps marked **(drv)** are driver-only.

---

## 1. Functional sequence

| # | Test | Pass criteria |
|--:|------|---------------|
| 1 | **(drv)** Driver loads | `pnputil /add-driver … /install` succeeds; no bugcheck; event log clean. |
| 2 | Device Manager shows camera | "RoyCam HD Camera" under Cameras; no ⚠ / code 10/52. |
| 3 | Windows Camera app opens it | Live frames (test pattern or fed); no freeze; correct resolution. |
| 4 | Privacy settings | Settings → Privacy → Camera allows desktop apps; camera accessible. |
| 5 | Chrome camera picker | "RoyCam HD Camera" listed and selectable. |
| 6 | WebRTC test page (`webcamtests.com` / `webrtc.github.io/samples`) | Frames render; resolution/fps reported sanely. |
| 7 | Zoom / Teams | Selectable, preview works, no crash on join/leave. |
| 8 | Target app (TikTok Live Studio) | Selectable, frames flow. |
| 9 | App close / reopen | Camera survives app exit → fallback frame; reopen resumes live frames. |
| 10 | Service restart | `sc stop/start RoyCamService` → fallback during gap → live resumes; no client crash. |
| 11 | **(drv)** Uninstall / reinstall | `pnputil /delete-driver … /uninstall` clean; reinstall works; no remnants. |
| 12 | **(drv)** Sleep / wake | S3/Modern-Standby → resume → camera still opens; no leaked resources. |
| 13 | Resolution change | Client switches 720p↔1080p↔480p → driver selects a supported media type cleanly. |
| 14 | FPS stability | Sustained 30fps (15fps mode = 15) within ±1; no creeping drift. |
| 15 | 1-hour endurance | No crash, no frame stall, bounded CPU/mem; error count 0. |
| 16 | 6-hour endurance (if 15 green) | No kernel nonpaged-pool growth, no handle/MMF growth. |

## 2. Metrics (capture for each run)

- **Actual FPS** (measure at the client, e.g. WebRTC stats / ffmpeg).
- **Frame drops** (gaps in `frame_index`).
- **Latency** app-produce → client-display (timestamp diff; target < 80 ms).
- **CPU%** (app, service, system) and **memory** (app, service working set).
- **Kernel nonpaged-pool** growth over time (`poolmon` / `!poolused`) — **(drv)**.
- **Camera open time** and **time-to-first-frame** (target < 1 s).
- **Error count** (service log + driver ETW).
- **Behavior on**: source disconnect, app crash (kill -9), service crash.

## 3. Fault-injection (must not crash anything)

| Inject | Expected |
|--------|----------|
| Kill app (`taskkill /f`) mid-stream | Service → fallback; driver shows fallback frame; clients keep running. |
| Kill service | Driver detects stale → fallback pattern; SCM restarts service; recovers. |
| Send oversized / garbage frame header | Service/driver **reject** (logged), no UB, stream continues. |
| Client opens camera before app starts | Fallback frame shown immediately. |
| Client requests unsupported resolution | Pin negotiates nearest supported media type; no failure. |
| Two clients open camera | v1: 2nd open fails **gracefully** (documented); no crash. |
| Rapid open/close loop (100×) | No handle/pool leak; stable. |

## 4. Driver-quality gates (track B) — **(drv)**

- **Static Driver Verifier (SDV)** + **Code Analysis** clean on the driver project.
- **Driver Verifier** (`verifier /standard /driver RoyCam.sys`) enabled during all
  endurance runs — no violations.
- **WDF/KMDF verifier** (if KMDF parts) on.
- WPP/ETW trace review: no user frame content logged; no error spam.

## 5. Automation (where practical)

- `tests/shared-memory-tests`: producer/consumer 30fps × 10 min, assert zero tears,
  monotonic index, latest-wins (see `ipc-frame-transport.md` §7).
- `tests/frame-pipeline-tests`: unit-test BGR→NV12/YUY2/RGB32 conversion vs reference
  (golden frames), stride/rounding correctness.
- `tests/service-tests`: install/start/feed/stop/uninstall smoke; status-pipe checks.
- `tests/endurance-tests`: scripted ffmpeg/WebRTC capture + metric logging for the
  1h/6h runs.
- Manual checklist for the GUI client steps (3,5,7,8) that can't be fully automated.

## 6. Sign-off criteria
- Steps 1–14 pass on a **clean Windows 11 x64** machine (and, for track B, on a
  **non-test-signing** machine once Microsoft-signed).
- 1h endurance green; 6h green for release.
- All fault-injection rows pass.
- Zero Driver Verifier / SDV violations (track B).

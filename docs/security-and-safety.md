# Security, Safety & Integrity

How this design honors the legal/safety constraints, and the system-stability rules
the implementation must follow. This is a **legitimate, transparent, own-branded
software camera** — nothing here deceives Windows, users, security software, or other
vendors.

---

## 1. Legal / integrity constraints — and how we meet them

| Constraint | How the design complies |
|------------|-------------------------|
| **No vendor impersonation** (Logitech/Microsoft/OBS/Elgato/etc.) | Device friendly name, INF, INF strings, and UI all say **"RoyCam HD Camera"** (our brand). No other vendor's name appears anywhere. |
| **No spoofed VID/PID / hardware ID / signature / certification** | Software camera uses **our own** hardware ID namespace (e.g. `ROOT\\RoyCamHDCamera` or our enumerated software device ID) — not a USB VID/PID, and never a third party's. Signed only with **our own** certificate via the proper Microsoft flow. |
| **No malware / stealth / anti-detection / bypass** | No hidden processes, no driver hiding, no tamper logic, no security-product evasion, no platform "is this virtual?" spoofing. The camera is openly a software source. |
| **Don't claim a software camera is physical USB hardware** | Docs, INF comments, and the README state plainly it is a **software-backed camera**. We do not fake USB descriptors or a physical bus. |
| **Legitimate custom device with our branding** | The entire product is ours; the value is exposing **our app's** frames as **our** camera. |
| **Follow Microsoft WDDM/driver practices** | AVStream minidriver modeled on Microsoft's AvsCamera sample; proper INF, KMDF/AVStream patterns, IRQL-correct code, WPP tracing, validated IOCTLs, Microsoft signing flow. |

> If at any point a requirement would require impersonation, ID spoofing, or
> detection evasion, the answer is **no** and the doc says so. The professional value
> here comes from a clean, honest, well-engineered device — not from pretending to be
> someone else's hardware.

## 2. Transparency & user control (no hidden capture)

- Camera output is **opt-in**: the user explicitly presses **Start Camera Output**.
  It is **off by default** (mirrors the existing `AVATAR_AICHART` / `AVATAR_TV`
  opt-in env gates).
- A clear **"● ON AIR / Camera Output Active"** indicator is shown while frames are
  being published, plus a **Stop** control.
- The camera advertises a **branded fallback frame** ("RoyCam HD Camera — waiting for
  source") rather than a silent black image, so a viewing app never shows an
  ambiguous frozen/black feed.
- **Logs record events and metrics only — never frame pixels or user content.** No
  screenshots, no payload bytes, no PII in logs.

## 3. Kernel-mode safety rules (driver)

The driver must **never** bugcheck Windows due to user-mode input or app/service
failure. Mandatory rules (enforced in review):

- **Trust nothing from user mode.** Validate every IOCTL: method, `InputBufferLength`,
  `OutputBufferLength`, and all header fields against the negotiated media type
  (see `ipc-frame-transport.md` §6).
- **Overflow-safe integer math** for all size/stride/offset computations
  (checked-multiply helpers; reject on overflow).
- **No per-frame allocation** in the streaming path; **preallocate** nonpaged buffers
  at pin-start, free at pin-stop.
- **No blocking** in the streaming/processing callback — bounded `memcpy` under a short
  lock only; no I/O, no waits, no pageable access at raised IRQL.
- **Bounded copies only** — copy `min(claimed, capacity)`, never the caller's length
  blindly.
- **Clean start/stop and power transitions**; release every resource on stop/remove;
  no leaks across cycles.
- **WPP/ETW tracing** for diagnostics; no `KdPrint` of user content.
- Keep the kernel **simple**: all colour conversion and heavy processing stay in user
  mode; the driver only paces output and copies already-formatted frames.

## 4. Service-mode safety rules

- Run as a **dedicated, least-privilege** service identity where feasible; request
  admin only for the explicit, user-initiated driver install/start operations.
- **Never load DLLs from writable/user-controlled directories** (set safe DLL search
  mode; load only from the install dir / System32). **Never execute user scripts.**
- Treat all app input as untrusted: validate dimensions/format/stride/size; **reject
  unsupported frames** before they reach the kernel.
- Handle app disconnect gracefully (→ fallback). Crash-isolate from the driver.
- Logs: events, counts, timings — **not** frame content.

## 5. App-mode safety rules

- No hardcoded absolute paths (use `%PUBLIC%` / install dir / config).
- Configurable camera name, resolution, fps; good, specific error messages.
- Non-blocking UI; frame production on a worker thread.
- The producer only writes to the shared ring and reads the status pipe — it has **no
  driver privileges** and cannot install/sign anything itself.

## 6. Distribution honesty (signing reality)

- Development uses **test signing** (dev machine only; visible watermark) — never a
  shipping mechanism.
- Production requires **our EV certificate + Microsoft Partner Center** attestation (or
  full WHQL/HLK) signing of the driver package. We will **not** ask end users to
  disable Secure Boot or enable test signing. If a clean signing path isn't available,
  we **ship the user-mode software camera** (MF/DShow, already built) instead of an
  unsigned kernel driver. Full detail in `driver-build-and-signing.md` (next).

## 7. Privacy posture summary

- Explicit start, visible indicator, easy stop.
- Branded, non-deceptive device.
- No content logging, no telemetry of frames.
- No background/auto capture, no admin-at-startup, no auto-loading driver tricks.

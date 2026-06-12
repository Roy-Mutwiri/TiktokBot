# Driver Build & Signing — the honest version

This is the document that determines whether the AVStream driver can actually reach
users. **Read it before investing in kernel code.** Nothing here hides complexity.

---

## 1. The one-sentence reality

> A kernel-mode driver will **not load on a normal Windows 10/11 machine** unless its
> driver package is **signed through Microsoft** (attestation or WHQL). Self-signed /
> test-signed drivers load **only** on machines you put into test-signing mode — that
> is a developer setting, not a distribution mechanism.

Everything below is about getting from "builds on my PC" to "loads on a user's PC."

## 2. Build prerequisites (dev machine)

| Need | This machine |
|------|--------------|
| Visual Studio 2022 / Build Tools | **Present** |
| Windows SDK 10.0.26100 | **Present** |
| **Windows Driver Kit (WDK) 10.0.26100** | **Installing now** (`Microsoft.WindowsWDK.10.0.26100`, matches the SDK) |
| WDK Visual Studio extension (driver templates/targets) | Comes with the WDK installer; verify after install |
| `signtool`, `makecert`/`pvk2pfx` or PowerShell `New-SelfSignedCertificate`, `inf2cat`, `stampinf` | From SDK/WDK |

WDK **must match** the SDK major version (both 26100) or driver builds fail.

## 3. Build

```powershell
# x64 release driver
msbuild driver\RoyCamDriver\RoyCamDriver.vcxproj /p:Configuration=Release /p:Platform=x64
# outputs: RoyCam.sys, RoyCam.inf, RoyCam.cat (after inf2cat)
```

## 4. Development signing (your dev box only)

```powershell
# 1) one-time: a self-signed TEST cert (clearly ours, e.g. CN=RoyCam Test)
New-SelfSignedCertificate -Type CodeSigningCert -Subject "CN=RoyCam Test Signing" `
  -CertStoreLocation Cert:\LocalMachine\My -KeyUsage DigitalSignature

# 2) build the catalog from the INF, then sign .sys and .cat with the test cert
inf2cat /driver:. /os:10_X64
signtool sign /fd SHA256 /a /s My /n "RoyCam Test Signing" RoyCam.cat
signtool sign /fd SHA256 /a /s My /n "RoyCam Test Signing" RoyCam.sys

# 3) trust the test cert on the dev machine (Root + TrustedPublisher)
# 4) enable test signing (DEV MACHINE ONLY — shows a desktop watermark)
bcdedit /set testsigning on        # reboot required
# 5) install
pnputil /add-driver RoyCam.inf /install
```

**Test signing caveats:** desktop watermark, weakens a security boundary, **off by
default**, and **blocked or awkward under Secure Boot**. Acceptable for a developer.
**Never** ask end users to do this.

## 5. Production signing (to ship to users)

Two Microsoft-blessed paths; both require enrolling our company:

### 5a. Attestation signing (recommended for a software camera)
1. **EV ("Extended Validation") code-signing certificate** from a CA (hardware token;
   ~1–3 day vetting; annual cost). Required to enroll.
2. **Microsoft Partner Center → Hardware program** (formerly Hardware Dev Center).
   Enroll the company using the EV cert.
3. Sign the driver package with the EV cert, submit the `.cab` for **attestation
   signing**. Microsoft returns a **Microsoft-signed** package that loads on
   Windows 10/11 (x64) without test-signing.
4. Distribute the Microsoft-signed package via our installer.

- Attestation = no hardware lab tests; fine for a software-only camera.
- Limitation: attestation-signed drivers target current Windows 10/11; not for older
  OSes and not the same as full certification.

### 5b. WHQL / HLK certification (heavier)
- Run the **Windows Hardware Lab Kit (HLK)** test suite against the driver, submit the
  HLK package, get a fully certified, "Windows Certified" signature.
- More work; needed only if we want the certification logo / Windows Update
  distribution. **Not required** for our use case.

## 6. Secure Boot & OS notes
- On Secure Boot machines, **only Microsoft-signed** kernel drivers load. So a shipped
  RoyCam driver **must** go through 5a/5b. Test-signing generally won't satisfy Secure
  Boot — another reason test-signing is dev-only.
- Target **Windows 10 1809+ / Windows 11 x64**. ARM64 is a separate signed build if
  ever needed.

## 7. Cost / timeline / friction (be candid with stakeholders)
- **EV certificate:** purchase + identity vetting (days), recurring annual cost,
  hardware token handling.
- **Partner Center enrollment:** company verification.
- **Per-update:** every driver change must be re-submitted/re-signed before users can
  install it — slower iteration than user-mode.
- **Support burden:** kernel driver install requires admin + can interact with AV/EDR
  software; more support cases than a user-mode camera.

## 8. The decision this forces

| Question | If yes | If no |
|----------|--------|-------|
| Do we have/obtain an EV cert + Partner Center now? | Proceed to ship the AVStream driver (attestation). | **Ship the user-mode camera** (`native_camera/` MF + `native_camera_dshow/` DShow) — branded, works today on normal machines, **no signing wall** — and keep the driver as a test-signed internal/dev build until enrollment is done. |

This is exactly why the roadmap is **dual-track**: the user-mode camera removes the
distribution risk *today*, while the AVStream driver matures behind the signing
process. We never ship an unsigned driver and never ask users to disable security.

## 9. Production checklist
- [ ] EV code-signing certificate acquired (hardware token).
- [ ] Microsoft Partner Center (Hardware) enrollment complete.
- [ ] Driver builds clean (Release/x64), passes static driver verifier (SDV) + Code
      Analysis.
- [ ] `.cab` submitted; **Microsoft-signed** package received.
- [ ] Installer ships the Microsoft-signed package; `pnputil` install verified on a
      **clean, non-test-signing** machine.
- [ ] Uninstall leaves no device/driver remnants (`uninstall-and-recovery.md`).
- [ ] Endurance + multi-app testing green (`testing-matrix.md`).

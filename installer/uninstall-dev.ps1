<#
  uninstall-dev.ps1 — fully reverse install-dev.ps1.

  Order (mirrors docs/uninstall-and-recovery.md app -> service -> driver):
    1. stop + unregister the RoyCam Frame Bridge scheduled task
    2. remove the shared ring directory (optional; -KeepData to retain)

  No driver is installed in Phase-1, so there is nothing to pnputil-remove yet.
#>
[CmdletBinding()]
param(
    [switch]$KeepData               # keep the ring directory
)

$ErrorActionPreference = "Continue"
$ringDir = "C:\Users\Public\RoyCamCamera"
$taskName = "RoyCam Frame Bridge (dev)"

try {
    Stop-ScheduledTask -TaskName $taskName -ErrorAction Stop
    Write-Host "[uninstall-dev] stopped task"
} catch { Write-Host "[uninstall-dev] task not running" }

try {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction Stop
    Write-Host "[uninstall-dev] unregistered task: $taskName"
} catch { Write-Host "[uninstall-dev] task not registered" }

if ($KeepData) {
    Write-Host "[uninstall-dev] -KeepData set; leaving $ringDir"
} elseif (Test-Path $ringDir) {
    try {
        Remove-Item -Path $ringDir -Recurse -Force -ErrorAction Stop
        Write-Host "[uninstall-dev] removed ring directory"
    } catch {
        Write-Host "[uninstall-dev] could not remove ring dir (in use?): $_"
    }
}

Write-Host "[uninstall-dev] done."

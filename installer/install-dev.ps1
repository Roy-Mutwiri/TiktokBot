<#
  install-dev.ps1 — Phase-1 (WDK-independent) dev setup for the RoyCam camera
  output pipeline.

  This does NOT install a kernel driver and does NOT require test-signing.
  It only:
    1. creates the shared ring directory under C:\Users\Public\RoyCamCamera
    2. registers a per-user Scheduled Task that runs the RoyCam Frame Bridge
       (service skeleton) at logon, so frames keep flowing without the UI.

  Everything here is additive and fully reversed by uninstall-dev.ps1.
  Run from an ordinary PowerShell (no admin needed for a per-user task).
#>
[CmdletBinding()]
param(
    [string]$PythonExe = "python",
    [switch]$NoTask                 # set up dirs only, don't register the task
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$ringDir = "C:\Users\Public\RoyCamCamera"
$taskName = "RoyCam Frame Bridge (dev)"

Write-Host "[install-dev] repo:    $repo"
Write-Host "[install-dev] ringDir: $ringDir"

if (-not (Test-Path $ringDir)) {
    New-Item -ItemType Directory -Path $ringDir -Force | Out-Null
    Write-Host "[install-dev] created ring directory"
} else {
    Write-Host "[install-dev] ring directory already exists"
}

if ($NoTask) {
    Write-Host "[install-dev] -NoTask set; skipping scheduled task"
    return
}

$svc = Join-Path $repo "service\roycam_service.py"
if (-not (Test-Path $svc)) { throw "service script not found: $svc" }

$action = New-ScheduledTaskAction -Execute $PythonExe `
    -Argument "`"$svc`"" -WorkingDirectory $repo
$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -StartWhenAvailable

try { Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction Stop } catch {}
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Settings $settings -Description "RoyCam Phase-1 user-mode frame bridge (dev)" | Out-Null

Write-Host "[install-dev] registered scheduled task: $taskName"
Write-Host "[install-dev] done. Start now with:"
Write-Host "    Start-ScheduledTask -TaskName `"$taskName`""
Write-Host "  or run the bridge directly:"
Write-Host "    $PythonExe `"$svc`" --preview"

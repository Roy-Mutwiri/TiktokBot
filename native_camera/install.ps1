# install.ps1 — register the media source DLL so the Frame Server can load it.
# Run as ADMINISTRATOR (it self-elevates). Pass -Uninstall to remove.
#
#   powershell -ExecutionPolicy Bypass -File native_camera\install.ps1
#   powershell -ExecutionPolicy Bypass -File native_camera\install.ps1 -Uninstall
#
param([switch]$Uninstall)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$dll  = Join-Path $here "VirtualCameraMediaSource.dll"

# self-elevate
$id = [Security.Principal.WindowsIdentity]::GetCurrent()
if (-not (New-Object Security.Principal.WindowsPrincipal($id)).IsInRole([Security.Principal.WindowsBuiltinRole]::Administrator)) {
    Write-Host "[install] elevating..."
    $a = "-ExecutionPolicy Bypass -File `"$($MyInvocation.MyCommand.Path)`""
    if ($Uninstall) { $a += " -Uninstall" }
    Start-Process powershell -Verb RunAs -ArgumentList $a
    return
}

if ($Uninstall) {
    Write-Host "[install] unregistering DLL..."
    & regsvr32 /u /s $dll
    Write-Host "[install] done. (The virtual camera only appears while avatar_camera.py --native runs anyway.)"
    return
}

if (-not (Test-Path $dll)) { throw "DLL not found: $dll — run build.ps1 first." }
Write-Host "[install] registering $dll ..."
& regsvr32 /s $dll
if ($LASTEXITCODE -ne 0) { throw "regsvr32 failed ($LASTEXITCODE)" }
Write-Host "[install] registered."
Write-Host "[install] Now run:  python avatar_camera.py --native"
Write-Host "[install] 'Avatar Studio Camera' will appear in apps while it runs, and vanish when you stop it."

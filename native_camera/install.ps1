# install.ps1 - register the Avatar Studio Camera media source so the Frame
# Server can instantiate it. The DLL has no DllRegisterServer (MF software
# sources register via a plain COM InprocServer32 entry), so we write the
# CLSID registry keys directly. Run as ADMINISTRATOR (it self-elevates).
#
#   powershell -ExecutionPolicy Bypass -File native_camera\install.ps1
#   powershell -ExecutionPolicy Bypass -File native_camera\install.ps1 -Uninstall
#
param([switch]$Uninstall)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$srcDll = Join-Path $here "VirtualCameraMediaSource.dll"
# The Windows Camera Frame Server (a service) cannot read DLLs from a user
# profile path, so install to a system location it CAN load.
$instDir = Join-Path $env:ProgramFiles "AvatarStudioCamera"
$dll  = Join-Path $instDir "VirtualCameraMediaSource.dll"
$clsid = "{7B89B92E-FE71-42D0-8A41-E137D06EA184}"
$key  = "HKLM:\SOFTWARE\Classes\CLSID\$clsid"

$id = [Security.Principal.WindowsIdentity]::GetCurrent()
if (-not (New-Object Security.Principal.WindowsPrincipal($id)).IsInRole([Security.Principal.WindowsBuiltinRole]::Administrator)) {
    Write-Host "[install] elevating..."
    $a = "-ExecutionPolicy Bypass -File `"$($MyInvocation.MyCommand.Path)`""
    if ($Uninstall) { $a += " -Uninstall" }
    Start-Process powershell -Verb RunAs -ArgumentList $a
    return
}

if ($Uninstall) {
    Write-Host "[install] removing CLSID registration..."
    Remove-Item -Path $key -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -Path $instDir -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "[install] done."
    return
}

if (-not (Test-Path $srcDll)) { throw "DLL not found: $srcDll - run build.ps1 first." }
New-Item -ItemType Directory -Force -Path $instDir | Out-Null
Copy-Item $srcDll $dll -Force
Write-Host "[install] installed DLL -> $dll"
Write-Host "[install] registering media source CLSID -> $dll"
New-Item -Path "$key\InprocServer32" -Force | Out-Null
New-ItemProperty -Path $key -Name "(default)" -Value "Avatar Studio Camera Media Source" -PropertyType String -Force | Out-Null
New-ItemProperty -Path "$key\InprocServer32" -Name "(default)" -Value $dll -PropertyType String -Force | Out-Null
New-ItemProperty -Path "$key\InprocServer32" -Name "ThreadingModel" -Value "Both" -PropertyType String -Force | Out-Null
Write-Host "[install] registered."
Write-Host "[install] Now run:  python avatar_camera.py --native"

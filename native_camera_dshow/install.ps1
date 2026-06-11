# install.ps1 - register the Avatar Studio Camera DirectShow filter.
# Run as ADMINISTRATOR (it self-elevates). Pass -Uninstall to remove.
#
# Registers by calling the DLL's own DllRegisterServer directly (more reliable
# than regsvr32 here). The filter joins the video-input-device category and
# appears as a normal webcam named "Avatar Studio Camera" - no virtual tag.
#
#   powershell -ExecutionPolicy Bypass -File native_camera_dshow\install.ps1
#   powershell -ExecutionPolicy Bypass -File native_camera_dshow\install.ps1 -Uninstall
#
param([switch]$Uninstall)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
# A DirectShow filter loads into the consuming app's process (which runs as the
# user), so it can be registered IN PLACE - no copy to a system folder needed.
$dll = Join-Path $here "AvatarCamFilter.dll"

$id = [Security.Principal.WindowsIdentity]::GetCurrent()
if (-not (New-Object Security.Principal.WindowsPrincipal($id)).IsInRole([Security.Principal.WindowsBuiltinRole]::Administrator)) {
    Write-Host "[install] elevating..."
    $a = "-ExecutionPolicy Bypass -File `"$($MyInvocation.MyCommand.Path)`""
    if ($Uninstall) { $a += " -Uninstall" }
    Start-Process powershell -Verb RunAs -ArgumentList $a
    return
}

$sig = @'
using System; using System.Runtime.InteropServices;
public class Reg {
  [DllImport("ole32.dll")] public static extern int CoInitialize(IntPtr p);
  [DllImport("kernel32", SetLastError=true)] public static extern IntPtr LoadLibrary(string p);
  [DllImport("kernel32")] public static extern bool FreeLibrary(IntPtr h);
  [DllImport("kernel32", SetLastError=true)] public static extern IntPtr GetProcAddress(IntPtr h, string n);
  [UnmanagedFunctionPointer(CallingConvention.StdCall)] public delegate int Fn();
}
'@
Add-Type $sig
[Reg]::CoInitialize([IntPtr]::Zero) | Out-Null
function Call-Export($path, $name) {
    $h = [Reg]::LoadLibrary($path); if ($h -eq [IntPtr]::Zero) { return -1 }
    $p = [Reg]::GetProcAddress($h, $name); if ($p -eq [IntPtr]::Zero) { [Reg]::FreeLibrary($h) | Out-Null; return -2 }
    $fn = [Runtime.InteropServices.Marshal]::GetDelegateForFunctionPointer($p, [Reg+Fn])
    $hr = $fn.Invoke(); [Reg]::FreeLibrary($h) | Out-Null; return $hr
}

if ($Uninstall) {
    # clean up any older Program Files registration too
    $oldDll = Join-Path $env:ProgramFiles "AvatarStudioCamera\AvatarCamFilter.dll"
    foreach ($d in @($dll, $oldDll)) { if (Test-Path $d) { Call-Export $d "DllUnregisterServer" | Out-Null } }
    Write-Host "[install] unregistered Avatar Studio Camera (DirectShow)."
    return
}

if (-not (Test-Path $dll)) { throw "DLL not found: $dll - run build.ps1 first." }
# remove a stale Program Files registration from an earlier install, if any
$oldDll = Join-Path $env:ProgramFiles "AvatarStudioCamera\AvatarCamFilter.dll"
if (Test-Path $oldDll) { Call-Export $oldDll "DllUnregisterServer" | Out-Null }
Call-Export $dll "DllUnregisterServer" | Out-Null   # idempotent re-register
$hr = Call-Export $dll "DllRegisterServer"
if ($hr -ne 0) { throw ("DllRegisterServer failed 0x{0:X8}" -f $hr) }
Write-Host "[install] registered 'Avatar Studio Camera' (DirectShow webcam, no virtual tag)."
Write-Host "[install] Feed it:  python avatar_camera.py --dshow"

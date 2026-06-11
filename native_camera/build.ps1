# build.ps1 — compile the Avatar Studio Camera media source DLL (x64 Release).
# Requires VS Build Tools + Windows SDK (see README). No admin needed to build.
#
#   powershell -ExecutionPolicy Bypass -File native_camera\build.ps1
#
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$proj = Join-Path $here "build\Windows-Camera\Samples\VirtualCamera\VirtualCameraMediaSource\VirtualCameraMediaSource.vcxproj"
if (-not (Test-Path $proj)) { throw "run setup.ps1 first (project not found: $proj)" }

# Locate MSBuild via vswhere (ships with any VS / Build Tools install).
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { throw "vswhere not found — is VS Build Tools installed? See README." }
$msbuild = & $vswhere -latest -requires Microsoft.Component.MSBuild -find "MSBuild\**\Bin\MSBuild.exe" | Select-Object -First 1
if (-not $msbuild) { throw "MSBuild not found — install the 'Desktop development with C++' workload." }
Write-Host "[build] using $msbuild"

# Restore NuGet packages the sample depends on (wil, cppwinrt).
$nuget = Get-Command nuget -ErrorAction SilentlyContinue
$sln = Join-Path $here "build\Windows-Camera\Samples\VirtualCamera\VirtualCameraSample.sln"
if ($nuget) {
    & $nuget.Source restore $sln
} else {
    Write-Host "[build] nuget.exe not on PATH — relying on MSBuild -restore."
}

& $msbuild $proj /t:Build /p:Configuration=Release /p:Platform=x64 /restore /m
if ($LASTEXITCODE -ne 0) { throw "build failed ($LASTEXITCODE)" }

$dll = Join-Path $here "build\Windows-Camera\Samples\VirtualCamera\x64\Release\VirtualCameraMediaSource.dll"
if (Test-Path $dll) {
    Copy-Item $dll $here -Force
    Write-Host "[build] DLL OK -> $(Join-Path $here 'VirtualCameraMediaSource.dll')"
} else {
    Write-Host "[build] build reported success but DLL not found at $dll — check output path."
}

# --- compile the session-lifetime host (vcam_host.exe) ----------------------
Write-Host "[build] compiling vcam_host.exe ..."
$devShellDll = & $vswhere -latest -find "Common7\Tools\Microsoft.VisualStudio.DevShell.dll" | Select-Object -First 1
$installPath = & $vswhere -latest -property installationPath
Import-Module $devShellDll
Enter-VsDevShell -VsInstallPath $installPath -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -no_logo" | Out-Null
Push-Location $here
& cl /nologo /EHsc /O2 /std:c++17 vcam_host.cpp `
    /Fe:vcam_host.exe `
    mfsensorgroup.lib mfplat.lib mfuuid.lib ole32.lib
$clrc = $LASTEXITCODE
Remove-Item -ErrorAction SilentlyContinue vcam_host.obj
Pop-Location
if ($clrc -eq 0 -and (Test-Path (Join-Path $here "vcam_host.exe"))) {
    Write-Host "[build] HOST OK -> $(Join-Path $here 'vcam_host.exe')"
    Write-Host "[build] Next (ADMIN):  native_camera\install.ps1"
} else {
    Write-Host "[build] host compile failed ($clrc)."
}

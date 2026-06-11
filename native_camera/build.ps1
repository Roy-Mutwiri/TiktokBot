# build.ps1 - compile the Avatar Studio Camera media source DLL (x64 Release).
# Requires VS Build Tools + Windows SDK (see README). No admin needed to build.
#
#   powershell -ExecutionPolicy Bypass -File native_camera\build.ps1
#
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$proj = Join-Path $here "build\Windows-Camera\Samples\VirtualCamera\VirtualCameraMediaSource\VirtualCameraMediaSource.vcxproj"
if (-not (Test-Path $proj)) { throw "run setup.ps1 first (project not found)" }

# Locate MSBuild via vswhere. -products * is REQUIRED to see Build Tools.
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { throw "vswhere not found - is VS Build Tools installed?" }
$msbuild = & $vswhere -latest -products * -requires Microsoft.Component.MSBuild -find "MSBuild\**\Bin\MSBuild.exe" | Select-Object -First 1
if (-not $msbuild) { throw "MSBuild not found - install the VCTools workload." }
Write-Host "[build] using $msbuild"

# Restore packages.config NuGet packages (CppWinRT, wil). MSBuild /restore does
# NOT handle packages.config, so we need nuget.exe; fetch it if absent.
$projDir = Split-Path $proj -Parent
$pkgCfg  = Join-Path $projDir "packages.config"
$nugetExe = (Get-Command nuget -ErrorAction SilentlyContinue).Source
if (-not $nugetExe) {
    $nugetExe = Join-Path $here "build\nuget.exe"
    if (-not (Test-Path $nugetExe)) {
        Write-Host "[build] downloading nuget.exe ..."
        Invoke-WebRequest "https://dist.nuget.org/win-x86-commandline/latest/nuget.exe" -OutFile $nugetExe
    }
}
if (Test-Path $pkgCfg) {
    Write-Host "[build] restoring NuGet packages..."
    & $nugetExe restore $pkgCfg -PackagesDirectory (Join-Path $projDir "packages")
    if ($LASTEXITCODE -ne 0) { throw "nuget restore failed ($LASTEXITCODE)" }
}

# Retarget to whatever Windows SDK is actually installed (the sample pins an
# older one). Pick the newest SDK that has the MF virtual camera header.
$sdkRoot = "C:\Program Files (x86)\Windows Kits\10\Include"
$sdkVer = Get-ChildItem $sdkRoot -Directory -ErrorAction SilentlyContinue |
    Where-Object { Test-Path (Join-Path $_.FullName "um\mfvirtualcamera.h") } |
    Sort-Object Name -Descending | Select-Object -First 1 -ExpandProperty Name
if (-not $sdkVer) { throw "no Windows SDK with mfvirtualcamera.h found" }
Write-Host "[build] retargeting to Windows SDK $sdkVer"

& $msbuild $proj /t:Build /p:Configuration=Release /p:Platform=x64 `
    /p:WindowsTargetPlatformVersion=$sdkVer /p:PlatformToolset=v143 /restore /m
if ($LASTEXITCODE -ne 0) { throw "DLL build failed ($LASTEXITCODE)" }

$dll = Get-ChildItem (Join-Path $here "build\Windows-Camera") -Recurse -Filter "VirtualCameraMediaSource.dll" -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -match "x64\\Release" } | Select-Object -First 1 -ExpandProperty FullName
if ($dll -and (Test-Path $dll)) {
    Copy-Item $dll $here -Force
    Write-Host "[build] DLL OK -> $(Join-Path $here 'VirtualCameraMediaSource.dll')"
} else {
    Write-Host "[build] DLL not found under build\ - check output path."
}

# --- compile the session-lifetime host (vcam_host.exe) ----------------------
Write-Host "[build] compiling vcam_host.exe ..."
$devShellDll = & $vswhere -latest -products * -find "Common7\Tools\Microsoft.VisualStudio.DevShell.dll" | Select-Object -First 1
$installPath = & $vswhere -latest -products * -property installationPath | Select-Object -First 1
Import-Module $devShellDll
Enter-VsDevShell -VsInstallPath $installPath -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -no_logo" | Out-Null
Push-Location $here
& cl /nologo /EHsc /O2 /std:c++17 vcam_host.cpp /Fe:vcam_host.exe mfsensorgroup.lib mfplat.lib mfuuid.lib ole32.lib
$clrc = $LASTEXITCODE
Remove-Item -ErrorAction SilentlyContinue vcam_host.obj
Pop-Location
if ($clrc -eq 0 -and (Test-Path (Join-Path $here "vcam_host.exe"))) {
    Write-Host "[build] HOST OK -> $(Join-Path $here 'vcam_host.exe')"
    Write-Host "[build] Next (ADMIN):  native_camera\install.ps1"
} else {
    Write-Host "[build] host compile failed ($clrc)."
}

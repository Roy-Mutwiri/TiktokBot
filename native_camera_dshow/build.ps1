# build.ps1 - compile the Avatar Studio Camera DirectShow filter (x64).
# Requires VS Build Tools + Windows SDK. No admin needed to build.
#
#   powershell -ExecutionPolicy Bypass -File native_camera_dshow\build.ps1
#
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$bc   = Join-Path $here "baseclasses"
if (-not (Test-Path $bc)) {
    Write-Host "[build] fetching DirectShow base classes..."
    $tmp = Join-Path $here "_classic"
    git clone --depth 1 --filter=blob:none --sparse https://github.com/microsoft/Windows-classic-samples.git $tmp
    Push-Location $tmp
    git sparse-checkout set Samples/Win7Samples/multimedia/directshow/baseclasses
    Pop-Location
    Copy-Item (Join-Path $tmp "Samples\Win7Samples\multimedia\directshow\baseclasses") $bc -Recurse -Force
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
    if (-not (Test-Path $bc)) { throw "could not fetch base classes" }
}

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$ip = & $vswhere -latest -products * -property installationPath | Select-Object -First 1
Import-Module (& $vswhere -latest -products * -find "Common7\Tools\Microsoft.VisualStudio.DevShell.dll" | Select-Object -First 1)
Enter-VsDevShell -VsInstallPath $ip -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -no_logo" | Out-Null

Push-Location $here
$defs = @('/D_UNICODE','/DUNICODE','/DWIN32','/D_WINDOWS','/D_CRT_SECURE_NO_WARNINGS')
$obj = Join-Path $here "obj"
New-Item -ItemType Directory -Force -Path $obj | Out-Null

# 1) compile the DirectShow base classes into strmbase.lib (static CRT /MT)
if (Test-Path (Join-Path $here "strmbase.lib")) {
    Write-Host "[build] strmbase.lib present - skipping base class rebuild (delete it to force)."
}
else {
Write-Host "[build] compiling base classes..."
$objfiles = @()
foreach ($f in Get-ChildItem "$bc\*.cpp") {
    $o = Join-Path $obj ($f.BaseName + ".obj")
    & cl /nologo /c /EHsc /O2 /MT /wd4996 /wd4995 $defs "/I$bc" /Fo"$o" $f.FullName 2>&1 |
        Where-Object { $_ -match "error|fatal" }
    if ($LASTEXITCODE -ne 0) { Pop-Location; throw "baseclass compile failed: $($f.Name)" }
    $objfiles += $o
}
& lib /nologo /out:strmbase.lib $objfiles | Out-Null
Write-Host "[build] strmbase.lib OK"
}

# 2) compile + link the filter DLL
Write-Host "[build] compiling AvatarCamFilter.dll..."
& cl /nologo /EHsc /O2 /MT /wd4996 $defs "/I$bc" /LD AvatarCamFilter.cpp `
    /Fe:AvatarCamFilter.dll `
    /link /DEF:AvatarCamFilter.def `
    strmbase.lib strmiids.lib winmm.lib ole32.lib oleaut32.lib uuid.lib `
    user32.lib advapi32.lib gdi32.lib
$rc = $LASTEXITCODE
Remove-Item -ErrorAction SilentlyContinue *.obj, AvatarCamFilter.exp, AvatarCamFilter.lib
Pop-Location
if ($rc -eq 0 -and (Test-Path (Join-Path $here "AvatarCamFilter.dll"))) {
    Write-Host "[build] OK -> $(Join-Path $here 'AvatarCamFilter.dll')"
    Write-Host "[build] Next (ADMIN):  native_camera_dshow\install.ps1"
} else {
    Write-Host "[build] filter link failed ($rc)."
}

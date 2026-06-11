# setup.ps1 - fetch Microsoft's VirtualCamera sample and graft our adaptation in.
# Run AFTER the toolchain is installed (see README). No admin needed for this.
#
#   powershell -ExecutionPolicy Bypass -File native_camera\setup.ps1
#
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$work = Join-Path $here "build"
$srcDir = Join-Path $work "Windows-Camera\Samples\VirtualCamera\VirtualCameraMediaSource"

New-Item -ItemType Directory -Force -Path $work | Out-Null

if (-not (Test-Path (Join-Path $work "Windows-Camera"))) {
    Write-Host "[setup] cloning Microsoft Windows-Camera sample (sparse)..."
    Push-Location $work
    git clone --depth 1 --filter=blob:none --sparse https://github.com/microsoft/Windows-Camera.git
    Push-Location "Windows-Camera"
    git sparse-checkout set Samples/VirtualCamera
    Pop-Location
    Pop-Location
} else {
    Write-Host "[setup] sample already present - skipping clone."
}

if (-not (Test-Path $srcDir)) { throw "media source dir not found: $srcDir" }

Write-Host "[setup] grafting Avatar Studio Camera files into the media source..."
Copy-Item (Join-Path $here "SharedFrame.h") $srcDir -Force
Copy-Item (Join-Path $here "SimpleFrameGenerator.cpp") $srcDir -Force

# Make the media source advertise 512x512 (the avatar frame size). The sample
# hard-codes its resolution as NUM_IMAGE_ROWS/COLS in SimpleMediaStream.cpp.
$sms = Join-Path $srcDir "SimpleMediaStream.cpp"
if (Test-Path $sms) {
    $txt = Get-Content $sms -Raw
    $new = $txt -replace '#define NUM_IMAGE_ROWS \d+', '#define NUM_IMAGE_ROWS 512' `
                -replace '#define NUM_IMAGE_COLS \d+', '#define NUM_IMAGE_COLS 512'
    if ($new -ne $txt) {
        Set-Content $sms $new -Encoding UTF8
        Write-Host "[setup] set media type to 512x512."
    } else {
        Write-Host "[setup] NOTE: NUM_IMAGE_ROWS/COLS not found - set 512x512 by hand."
    }
}

# The sample hand-defines two GUIDs that the modern SDK (>=22621) now provides,
# causing C2374 redefinition. Comment them out (the SDK supplies them).
$vh = Join-Path $srcDir "VirtualCameraMediaSource.h"
if (Test-Path $vh) {
    $txt = Get-Content $vh -Raw
    $new = $txt -replace '(?s)DEFINE_GUID\(MF_VIRTUALCAMERA_PROVIDE_ASSOCIATED_CAMERA_SOURCES.*?\);', '/* provided by SDK */' `
                -replace '(?s)DEFINE_GUID\(MF_VIRTUALCAMERA_ASSOCIATED_CAMERA_SOURCES.*?\);', '/* provided by SDK */'
    if ($new -ne $txt) {
        Set-Content $vh $new -Encoding UTF8
        Write-Host "[setup] removed duplicate MF_VIRTUALCAMERA GUID definitions."
    }
}

Write-Host "[setup] done. Next:  native_camera\build.ps1"

# setup.ps1 — fetch Microsoft's VirtualCamera sample and graft our adaptation in.
# Run AFTER the toolchain is installed (see README). No admin needed for this.
#
#   powershell -ExecutionPolicy Bypass -File native_camera\setup.ps1
#
$ErrorActionPreference = "Stop"
$here  = Split-Path -Parent $MyInvocation.MyCommand.Path
$work  = Join-Path $here "build"
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
    Write-Host "[setup] sample already present — skipping clone."
}

if (-not (Test-Path $srcDir)) { throw "media source dir not found: $srcDir" }

Write-Host "[setup] grafting Avatar Studio Camera files into the media source..."
Copy-Item (Join-Path $here "SharedFrame.h")            $srcDir -Force
Copy-Item (Join-Path $here "SimpleFrameGenerator.cpp") $srcDir -Force

# Make the media source advertise 512x512 (the avatar frame size). The sample
# hard-codes its resolutions in SimpleMediaSource.cpp; patch the first WxH pair.
$sms = Join-Path $srcDir "SimpleMediaSource.cpp"
if (Test-Path $sms) {
    $txt = Get-Content $sms -Raw
    # Common patterns in the sample: MFSetAttributeSize(... MF_MT_FRAME_SIZE, W, H)
    $patched = [regex]::Replace($txt,
        'MF_MT_FRAME_SIZE,\s*\d+,\s*\d+',
        'MF_MT_FRAME_SIZE, 512, 512', 1)
    if ($patched -ne $txt) {
        Set-Content $sms $patched -Encoding UTF8
        Write-Host "[setup] set media type to 512x512."
    } else {
        Write-Host "[setup] NOTE: could not auto-patch resolution — set 512x512 in SimpleMediaSource.cpp by hand."
    }
}

Write-Host "[setup] done. Next:  native_camera\build.ps1"

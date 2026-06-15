[CmdletBinding()]
param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repo

& python -c "import PyInstaller" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller is missing. Install it with: python -m pip install pyinstaller"
}

$buildArgs = @("-m", "PyInstaller", "--noconfirm")
if ($Clean) {
    $buildArgs += "--clean"
}
$buildArgs += "AvatarStudio.spec"

& python @buildArgs
if ($LASTEXITCODE -ne 0) {
    throw "Avatar Studio executable build failed."
}

$exe = Join-Path $repo "dist\AvatarStudio\AvatarStudio.exe"
if (-not (Test-Path $exe)) {
    throw "Build completed without producing $exe"
}

Write-Host "Built: $exe"

[CmdletBinding()]
param(
    [switch]$Clean,
    [switch]$SkipAppBuild
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repo

if (-not $SkipAppBuild) {
    $buildArgs = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", (Join-Path $repo "build_exe.ps1")
    )
    if ($Clean) {
        $buildArgs += "-Clean"
    }
    & powershell @buildArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Avatar Studio application build failed."
    }
}

$appExe = Join-Path $repo "dist\AvatarStudio\AvatarStudio.exe"
if (-not (Test-Path $appExe)) {
    throw "Application package is missing: $appExe"
}

$appDir = Join-Path $repo "dist\AvatarStudio"
$appBytes = (
    Get-ChildItem $appDir -Recurse -File |
    Measure-Object Length -Sum
).Sum
$driveRoot = [IO.Path]::GetPathRoot($repo)
$freeBytes = [IO.DriveInfo]::new($driveRoot).AvailableFreeSpace
$minimumFree = 12GB
if ($freeBytes -lt $minimumFree) {
    throw (
        "At least 12 GB of free working space is required before compiling " +
        "the installer. Available: {0:N2} GB." -f ($freeBytes / 1GB)
    )
}

$isccCandidates = @(
    (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe"),
    (Join-Path $env:ProgramFiles "Inno Setup 6\ISCC.exe"),
    (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe")
)
$iscc = $isccCandidates | Where-Object {
    $_ -and (Test-Path -LiteralPath $_)
} | Select-Object -First 1

if (-not $iscc) {
    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "Inno Setup 6 is required to build the installer."
    }
    Write-Host "Installing Inno Setup 6 on this build PC..."
    & $winget.Source install --id JRSoftware.InnoSetup --exact `
        --silent --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        throw "Inno Setup installation failed."
    }
    $iscc = $isccCandidates | Where-Object {
        $_ -and (Test-Path -LiteralPath $_)
    } | Select-Object -First 1
}

if (-not $iscc) {
    throw "Inno Setup installed but ISCC.exe could not be located."
}

$outputDir = Join-Path $repo "dist\AvatarStudio-Installer"
if (Test-Path $outputDir) {
    $resolved = (Resolve-Path -LiteralPath $outputDir).Path
    if (-not $resolved.StartsWith(
            $repo + [IO.Path]::DirectorySeparatorChar,
            [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to clean output outside the repository: $resolved"
    }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}

& $iscc (Join-Path $repo "installer\AvatarStudio.iss")
if ($LASTEXITCODE -ne 0) {
    throw "Avatar Studio installer build failed."
}

$setup = Join-Path $outputDir "AvatarStudio-Setup.exe"
if (-not (Test-Path $setup)) {
    throw "Installer build completed without producing $setup"
}

$sharingGuide = Join-Path $repo "installer\SHARING.txt"
Copy-Item -LiteralPath $sharingGuide -Destination $outputDir -Force
$hashes = Get-ChildItem $outputDir -File |
    Where-Object { $_.Name -ne "SHA256SUMS.txt" } |
    Get-FileHash -Algorithm SHA256
$hashLines = $hashes | ForEach-Object {
    "{0}  {1}" -f $_.Hash, (Split-Path $_.Path -Leaf)
}
Set-Content -LiteralPath (Join-Path $outputDir "SHA256SUMS.txt") `
    -Value $hashLines -Encoding ascii

$files = Get-ChildItem $outputDir -File
$total = ($files | Measure-Object Length -Sum).Sum
Write-Host ""
Write-Host "Installer ready:"
Write-Host "  $setup"
Write-Host ("  {0} files, {1:N2} GB total" -f $files.Count, ($total / 1GB))
Write-Host ""
Write-Host "Share the entire AvatarStudio-Installer folder."
Write-Host "The recipient only double-clicks AvatarStudio-Setup.exe."

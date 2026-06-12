<#
  install.ps1 — install the (test-signed) RoyCam HD Camera driver as a
  root-enumerated software device. ADMIN + test-signing enabled required.

  Honest preconditions (the script checks them and refuses otherwise):
    * RoyCamDriver.sys/.inf/.cat present (build + sign first)
    * test-signing ON (bcdedit) for a self-signed dev cert to load
#>
[CmdletBinding()]
param(
    [ValidateSet("Debug","Release")] [string]$Config = "Release"
)

$ErrorActionPreference = "Stop"
$outDir = Join-Path $PSScriptRoot "RoyCamDriver\x64\$Config\RoyCamDriver"
$inf = Join-Path $outDir "RoyCamDriver.inf"

if (-not (Test-Path $inf)) { throw "not built: $inf not found (run build.ps1 + sign-testcert.ps1)" }

# Warn if test-signing is off (self-signed driver will fail to load).
$ts = (bcdedit /enum '{current}' | Select-String -SimpleMatch "testsigning")
if (-not $ts -or ($ts -notmatch "Yes")) {
    Write-Host "[install] WARNING: test-signing appears OFF. A dev-signed driver"
    Write-Host "          will not load until:  bcdedit /set testsigning on  + reboot"
}

Write-Host "[install] pnputil add+install: $inf"
pnputil /add-driver $inf /install

Write-Host "[install] creating root-enumerated device node (root\RoyCamHDCamera)"
# devgen ships with the WDK; falls back to a note if absent.
$devgen = "C:\Program Files (x86)\Windows Kits\10\Tools\10.0.26100.0\x64\devgen.exe"
if (Test-Path $devgen) {
    & $devgen /add /hardwareid "root\RoyCamHDCamera"
} else {
    Write-Host "[install] devgen not found; add the device via Device Manager ->"
    Write-Host "          Add legacy hardware -> have disk -> RoyCamDriver.inf"
}
Write-Host "[install] done. Check Device Manager -> Cameras -> RoyCam HD Camera."

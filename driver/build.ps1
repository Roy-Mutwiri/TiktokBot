<#
  build.ps1 — build the RoyCam HD Camera driver (.sys + .inf) with the WDK.

  Requires the VS<->WDK integration ("WindowsKernelModeDriver10.0" platform
  toolset). The WDK headers/libs/build-props from
  `winget install Microsoft.WindowsWDK.10.0.26100` are necessary but NOT
  sufficient on a Build Tools-only box: the toolset wrapper ships in the WDK
  Visual Studio extension. If msbuild reports MSB8020, install that extension
  (see driver/README.md "Build prerequisites") and re-run.

  Output: driver\RoyCamDriver\x64\<Config>\RoyCamDriver\ (RoyCamDriver.sys/.inf)
#>
[CmdletBinding()]
param(
    [ValidateSet("Debug","Release")] [string]$Config = "Release",
    [string]$MSBuild = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\MSBuild.exe"
)

$ErrorActionPreference = "Stop"
$proj = Join-Path $PSScriptRoot "RoyCamDriver\RoyCamDriver.vcxproj"

if (-not (Test-Path $MSBuild)) { throw "MSBuild not found at $MSBuild" }
if (-not (Test-Path $proj))    { throw "project not found: $proj" }

Write-Host "[build] $Config|x64  ($proj)"
& $MSBuild $proj /p:Configuration=$Config /p:Platform=x64 /v:minimal /nologo
if ($LASTEXITCODE -ne 0) {
    Write-Host "[build] FAILED (exit $LASTEXITCODE)."
    Write-Host "[build] If this is MSB8020, the WDK VS2022 extension (kernel-mode"
    Write-Host "        toolset) is not installed into Build Tools. See README.md."
    exit $LASTEXITCODE
}

$out = Join-Path $PSScriptRoot "RoyCamDriver\x64\$Config\RoyCamDriver"
Write-Host "[build] OK -> $out"
Get-ChildItem $out -ErrorAction SilentlyContinue | Select-Object Name,Length

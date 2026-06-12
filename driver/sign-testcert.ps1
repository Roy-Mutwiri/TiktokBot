<#
  sign-testcert.ps1 — DEV-ONLY test signing for the RoyCam driver.

  Creates a self-signed test certificate, signs RoyCamDriver.sys + .cat with it,
  and installs it into the local test-root stores. This is for a DEVELOPER box
  with test-signing enabled ONLY. It is NOT production signing and the OS shows
  a "Test Mode" watermark. Production distribution needs an EV certificate plus
  Microsoft Partner Center attestation (see docs/driver-build-and-signing.md).

  This script does NOT enable test-signing or reboot — that is a separate,
  explicit step the operator performs:  bcdedit /set testsigning on  (+ reboot).
#>
[CmdletBinding()]
param(
    [ValidateSet("Debug","Release")] [string]$Config = "Release",
    [string]$CertName = "RoyCam Test Cert (DEV ONLY)",
    [string]$Bin = "C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64"
)

$ErrorActionPreference = "Stop"
$outDir = Join-Path $PSScriptRoot "RoyCamDriver\x64\$Config\RoyCamDriver"
$sys = Join-Path $outDir "RoyCamDriver.sys"
$cat = Join-Path $outDir "RoyCamDriver.cat"
$signtool = Join-Path $Bin "signtool.exe"

if (-not (Test-Path $sys)) { throw "build first: $sys not found" }
if (-not (Test-Path $signtool)) { throw "signtool not found: $signtool" }

# Reuse an existing test cert or make one.
$cert = Get-ChildItem Cert:\CurrentUser\My | Where-Object { $_.Subject -eq "CN=$CertName" } | Select-Object -First 1
if ($null -eq $cert) {
    Write-Host "[sign] creating self-signed test cert: $CertName"
    $cert = New-SelfSignedCertificate -Type CodeSigningCert -Subject "CN=$CertName" `
        -CertStoreLocation Cert:\CurrentUser\My -KeyUsage DigitalSignature `
        -TextExtension @("2.5.29.37={text}1.3.6.1.5.5.7.3.3")
}
$thumb = $cert.Thumbprint
Write-Host "[sign] thumbprint $thumb"

# Trust it on THIS dev box (test root + trusted publisher).
$tmpCer = Join-Path $env:TEMP "roycam_testcert.cer"
Export-Certificate -Cert $cert -FilePath $tmpCer | Out-Null
certutil -addstore -f Root $tmpCer | Out-Null
certutil -addstore -f TrustedPublisher $tmpCer | Out-Null

Write-Host "[sign] signing $sys"
& $signtool sign /fd SHA256 /sha1 $thumb /t http://timestamp.digicert.com $sys
if (Test-Path $cat) {
    Write-Host "[sign] signing $cat"
    & $signtool sign /fd SHA256 /sha1 $thumb /t http://timestamp.digicert.com $cat
} else {
    Write-Host "[sign] NOTE: no .cat yet (inf2cat runs during build); sign it after."
}
Write-Host "[sign] done. Reminder: enable test-signing + reboot before loading:"
Write-Host "        bcdedit /set testsigning on   (admin, then reboot)"

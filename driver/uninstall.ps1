<#
  uninstall.ps1 — remove the RoyCam HD Camera device + driver. ADMIN.
  Reverses install.ps1. Order: remove device node -> delete-driver from store.
#>
[CmdletBinding()]
param()

$ErrorActionPreference = "Continue"

# Remove the root-enumerated device node(s).
$devgen = "C:\Program Files (x86)\Windows Kits\10\Tools\10.0.26100.0\x64\devgen.exe"
Write-Host "[uninstall] removing root\RoyCamHDCamera device node(s)"
try {
    $ids = (pnputil /enum-devices /class Camera) 2>$null
    # Best-effort: use devcon/pnputil to remove by hardware id if present.
    pnputil /remove-device /deviceid "root\RoyCamHDCamera" 2>$null
} catch { Write-Host "[uninstall] device node removal skipped: $_" }

# Find and delete the published driver package (oemNN.inf) for RoyCam.
Write-Host "[uninstall] locating published RoyCam driver package"
$pkgs = pnputil /enum-drivers
$oem = $null
for ($i = 0; $i -lt $pkgs.Count; $i++) {
    if ($pkgs[$i] -match "Original Name:\s*RoyCamDriver.inf") {
        for ($j = $i; $j -ge 0; $j--) {
            if ($pkgs[$j] -match "Published Name:\s*(oem\d+\.inf)") { $oem = $Matches[1]; break }
        }
        break
    }
}
if ($oem) {
    Write-Host "[uninstall] pnputil /delete-driver $oem /uninstall /force"
    pnputil /delete-driver $oem /uninstall /force
} else {
    Write-Host "[uninstall] RoyCamDriver.inf not found in the driver store (already removed?)"
}
Write-Host "[uninstall] done."

$ErrorActionPreference = "Continue"

$project = Split-Path -Parent $MyInvocation.MyCommand.Path
$script = Join-Path $project "avatar_studio.py"
$log = Join-Path $project "restart_avatar_studio.log"
$pidFile = Join-Path $project "avatar_studio.pid"

Add-Content -Path $log -Value "$(Get-Date -Format o) restart requested"

$oldPid = $null
if (Test-Path $pidFile) {
    try {
        $oldPid = [int](Get-Content $pidFile -ErrorAction Stop | Select-Object -First 1)
    } catch {
        $oldPid = $null
    }
}
if ($oldPid) {
    $oldProc = Get-Process -Id $oldPid -ErrorAction SilentlyContinue
    if ($oldProc) {
        Add-Content -Path $log -Value "$(Get-Date -Format o) stopping pid=$oldPid"
        Stop-Process -Id $oldPid -Force -ErrorAction SilentlyContinue
    }
}

$env:TCL_LIBRARY = "//?/C:/Users/user/AppData/Local/Programs/Python/Python311/tcl/tcl8.6"
$env:TK_LIBRARY = "//?/C:/Users/user/AppData/Local/Programs/Python/Python311/tcl/tk8.6"
$env:AVATAR_AUTOCONFIG = "0"

Set-Location $project
Add-Content -Path $log -Value "$(Get-Date -Format o) starting fresh UI"
$proc = Start-Process python -ArgumentList "`"$script`"" -WorkingDirectory $project -PassThru
Set-Content -Path $pidFile -Value $proc.Id
Add-Content -Path $log -Value "$(Get-Date -Format o) started pid=$($proc.Id)"
$proc.WaitForExit()
Add-Content -Path $log -Value "$(Get-Date -Format o) exited code=$LASTEXITCODE"

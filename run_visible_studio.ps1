$env:TCL_LIBRARY = "//?/C:/Users/user/AppData/Local/Programs/Python/Python311/tcl/tcl8.6"
$env:TK_LIBRARY = "//?/C:/Users/user/AppData/Local/Programs/Python/Python311/tcl/tk8.6"
$env:AVATAR_AUTOCONFIG = "0"
Set-Location "C:\Users\user\Documents\TiktokBot"
Add-Content -Path "run_visible_studio.log" -Value "$(Get-Date -Format o) starting python"
python avatar_studio.py
Add-Content -Path "run_visible_studio.log" -Value "$(Get-Date -Format o) python exited code=$LASTEXITCODE"

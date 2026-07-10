$env:TCL_LIBRARY = "//?/C:/Users/user/AppData/Local/Programs/Python/Python311/tcl/tcl8.6"
$env:TK_LIBRARY = "//?/C:/Users/user/AppData/Local/Programs/Python/Python311/tcl/tk8.6"
Set-Location "C:\Users\user\Documents\TiktokBot"
Add-Content -Path "_studio_launcher.log" -Value "$(Get-Date -Format o) launcher starting"
python avatar_studio.py
Add-Content -Path "_studio_launcher.log" -Value "$(Get-Date -Format o) launcher exited code=$LASTEXITCODE"

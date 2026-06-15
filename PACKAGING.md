# Windows installer

Avatar Studio is packaged as a self-contained PyInstaller application and then
wrapped in a normal Windows installer. The destination PC does not need Python,
pip, FFmpeg, PyTorch, or Playwright installed.

## Build the shareable installer

```powershell
python -m pip install pyinstaller
powershell -NoProfile -ExecutionPolicy Bypass -File .\build_installer.ps1 -Clean
```

The installer is written to:

```text
dist\AvatarStudio-Installer\AvatarStudio-Setup.exe
```

Share the complete `dist\AvatarStudio-Installer` folder. The recipient only
double-clicks `AvatarStudio-Setup.exe`. Setup installs under the current user's
`AppData\Local\Programs` directory, creates Start Menu and Desktop shortcuts,
and offers to launch the program.

The package includes Python, FFmpeg, PyTorch/CUDA runtime libraries, Chromium
for Playwright, application assets, voices, music, and local model files.

System-level capabilities still depend on the PC:

- Windows 10 1809 or newer, 64-bit
- A compatible graphics driver for GPU acceleration; CPU fallback remains
- OBS Virtual Camera or another supported virtual-camera backend for camera
  output
- Ollama only when the local AI brain feature is desired

Because the offline package is very large, Inno Setup may create `.bin` data
files beside the setup executable. Keep those files in the same folder as
`AvatarStudio-Setup.exe`.

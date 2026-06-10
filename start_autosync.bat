@echo off
REM ===========================================================================
REM start_autosync.bat - run the auto-commit/auto-push watcher in the background
REM Double-click to start it; it keeps running (hidden) and pushes every change.
REM Stop it via Task Manager (pythonw.exe) or the Stop-AutoSync note in README.
REM ===========================================================================
cd /d "%~dp0"
start "TiktokBot autosync" /min pythonw "%~dp0autosync.py"
echo Auto-sync watcher started in the background.
echo It commits and pushes to GitHub a few seconds after any change.
echo Activity log: autosync.log

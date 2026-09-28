@echo off
rem Sahaay launcher for Windows on Snapdragon. Uses the installer's virtual environment when present.
setlocal
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m sahaay %*
) else (
    where py >nul 2>nul && (py -3.12 -m sahaay %*) || (python -m sahaay %*)
)
endlocal

@echo off
rem Sahaay launcher for Windows on Snapdragon (ARM64 Python 3.12)
setlocal
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
where py >nul 2>nul && (py -3.12 -m sahaay %*) || (python -m sahaay %*)
endlocal

@echo off
setlocal
cd /d "%~dp0"

if not exist output mkdir output
if not exist input mkdir input
if not exist models mkdir models

echo Starting ComputerVision Counter Images...
echo.

where py >nul 2>nul
if %errorlevel%==0 (
    py -3.12 bootstrap_env.py || py bootstrap_env.py
    py -3.12 start_app.py || py start_app.py
) else (
    python bootstrap_env.py
    python start_app.py
)

echo.
pause

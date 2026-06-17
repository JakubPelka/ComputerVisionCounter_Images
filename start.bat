@echo off
setlocal
cd /d "%~dp0"

if not exist output mkdir output
if not exist input mkdir input
if not exist models mkdir models

echo Starting ComputerVision Counter Images...
echo.

:: Ensure virtual environment exists
if not exist .venv (
    echo [INFO] Creating virtual environment...
    where py >nul 2>nul
    if %errorlevel%==0 (
        py -3 -m venv .venv
    ) else (
        python -m venv .venv
    )
)

:: Activate and install requirements
call .venv\Scripts\activate.bat
echo [INFO] Updating dependencies...
pip install -r requirements.txt --quiet

:: Run the application
python src\start_app.py

echo.
pause

@echo off
setlocal
set "BASE=%~dp0"
set "PYTHONNOUSERSITE=1"

rem Try system Python first, then the Windows launcher
where python >nul 2>&1 && (set "PYEXE=python") || (where py >nul 2>&1 && (set "PYEXE=py -3"))

if not defined PYEXE (
  echo [ERROR] Python 3.10+ is required. Please install it from https://www.python.org/downloads/
  echo Then double-click this file again.
  pause
  exit /b 1
)

rem One command handles first-run and subsequent runs:
rem - bootstrap_env installs/updates into .\_pkgs, enforces local-only imports, then launches the app.
"%PYEXE%" "%BASE%bootstrap_env.py" "%BASE%start_app.py"
echo.
pause

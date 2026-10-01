@echo off
REM ---------------------------------------------------------------------------
REM  PI Detection - start the Pipeline Wizard (Windows)
REM
REM  Activates the project virtual environment (venv\) and launches the UI.
REM  Double-click this file, or run it from any directory.
REM
REM  Extra arguments are forwarded to the target, e.g.  start_ui_win.bat --help
REM  Set UI_TARGET to run something else with the venv active, e.g.
REM      set UI_TARGET=train.py   && start_ui_win.bat
REM ---------------------------------------------------------------------------
setlocal

REM Run from the folder this script lives in, whatever the current directory is.
cd /d "%~dp0"

if not defined UI_TARGET set "UI_TARGET=pipeline_ui.py"

set "VENV_ACT=venv\Scripts\activate.bat"
set "VENV_PY=venv\Scripts\python.exe"

if not exist "%VENV_ACT%" (
    echo.
    echo [ERROR] Virtual environment not found ^(expected "%VENV_ACT%"^).
    echo.
    echo         Create it once from this folder:
    echo             python -m venv venv
    echo             venv\Scripts\activate
    echo             pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo Activating virtual environment ...
call "%VENV_ACT%"

REM Prefer the activated interpreter; fall back to the venv python directly.
set "PY=python"
where python >nul 2>nul || set "PY=%VENV_PY%"

echo Starting %UI_TARGET% ...
"%PY%" "%UI_TARGET%" %*
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
    echo.
    echo [%UI_TARGET% exited with code %RC%]
    pause
)
exit /b %RC%

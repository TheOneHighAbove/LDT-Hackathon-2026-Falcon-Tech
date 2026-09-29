@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if exist ".venv\Scripts\pythonw.exe" goto :launch

echo Falcon ReID: preparing the optional GUI environment...
where py >nul 2>nul
if not errorlevel 1 (
  py -3 -m venv .venv
) else (
  where python >nul 2>nul
  if errorlevel 1 goto :no_python
  python -m venv .venv
)
if errorlevel 1 goto :setup_failed

".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements-gui.txt
if errorlevel 1 goto :setup_failed

".venv\Scripts\python.exe" -c "import tkinter; from PIL import Image"
if errorlevel 1 goto :setup_failed

:launch
start "Falcon ReID" ".venv\Scripts\pythonw.exe" -m prototype.desktop_app
exit /b 0

:no_python
echo Python 3 was not found. Install Python 3.11 or newer and run this file again.
pause
exit /b 1

:setup_failed
echo Falcon ReID: GUI environment setup failed. See the error above.
pause
exit /b 1

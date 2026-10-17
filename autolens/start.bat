@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    py -3.12 -m venv .venv
    if errorlevel 1 goto error
)
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto error
.venv\Scripts\python.exe run.py
pause
exit /b
:error
echo Setup failed. Install Python 3.12, or follow the manual commands in README.md.
pause
exit /b 1

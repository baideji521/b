@echo off
chcp 65001 >nul
setlocal
set "ROOT=%~dp0"
cd /d "%ROOT%"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONPATH=%ROOT%src"
set "VENV_PY=%ROOT%.venv\Scripts\pythonw.exe"
if not exist "%VENV_PY%" set "VENV_PY=%ROOT%.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo [FAIL] 还没有虚拟环境，请先运行 setup_and_test.bat
    pause
    exit /b 1
)

start "" "%VENV_PY%" "%ROOT%run.py" gui %*
exit /b 0

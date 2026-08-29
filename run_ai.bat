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

rem 只开 AI 面板（第二主界面）：AI 设置 + 自动剪辑。加 --auto 就开起来直接跑
start "" "%VENV_PY%" "%ROOT%run.py" ai %*
exit /b 0

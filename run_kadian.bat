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

rem AI_卡点舞：独立界面。目标歌对齐 → 固定位置切片 → 素材库 → 多版本混剪
rem 和主界面各开各的，两边互不影响，可以同时开着
start "" "%VENV_PY%" "%ROOT%run.py" dance-montage gui %*
exit /b 0

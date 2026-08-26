@echo off
chcp 65001 >nul
setlocal
set "ROOT=%~dp0"
cd /d "%ROOT%"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONPATH=%ROOT%src"
set "VENV_PY=%ROOT%.venv\Scripts\python.exe"

echo ============================================================
echo  无人值守全流程：环境 -^> 依赖 -^> 模型 -^> 冒烟 -^> 视频分析 -^> 报告
echo  开始时间 %DATE% %TIME%
echo ============================================================

if not exist "%ROOT%input" mkdir "%ROOT%input"

if not exist "%VENV_PY%" (
    echo [venv] 创建虚拟环境 .venv
    py -3 -m venv "%ROOT%.venv"
    if not exist "%VENV_PY%" python -m venv "%ROOT%.venv"
)
if not exist "%VENV_PY%" (
    echo [FAIL] 无法创建虚拟环境，请确认已安装 Python 3.10+ 并加入 PATH
    exit /b 1
)

"%VENV_PY%" "%ROOT%tools\bootstrap.py" --all --log-name auto
set "RC=%ERRORLEVEL%"

echo ============================================================
echo  结束时间 %DATE% %TIME%
echo ============================================================
if exist "%ROOT%FINAL_REPORT.txt" (
    echo.
    type "%ROOT%FINAL_REPORT.txt"
)
echo.
if "%RC%"=="0" (
    echo [OK] 全流程完成。详见 FINAL_REPORT.txt 与 output\ 目录
) else (
    echo [FAIL] 存在失败步骤。详见 FINAL_REPORT.txt 与 logs\ 目录
)
exit /b %RC%

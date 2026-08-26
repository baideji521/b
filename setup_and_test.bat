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
echo  安装与自检: Python / GPU / CUDA / 依赖 / 模型 / 冒烟测试
echo ============================================================

if not exist "%VENV_PY%" (
    echo [venv] 创建虚拟环境 .venv
    py -3 -m venv "%ROOT%.venv"
    if not exist "%VENV_PY%" python -m venv "%ROOT%.venv"
)
if not exist "%VENV_PY%" (
    echo [FAIL] 无法创建虚拟环境，请确认已安装 Python 3.10+ 并加入 PATH
    exit /b 1
)

"%VENV_PY%" "%ROOT%tools\bootstrap.py" --install --verify --download --smoke --log-name setup
set "RC=%ERRORLEVEL%"

echo ============================================================
if "%RC%"=="0" (
    echo [OK] 安装与自检完成。接下来可以直接运行 run_auto.bat
) else (
    echo [FAIL] 安装或自检失败，详见 logs\ 目录
)
exit /b %RC%

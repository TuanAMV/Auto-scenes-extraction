@echo off
setlocal
chcp 65001 >nul

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHON_EXE=%~dp0python\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Embedded Python not found: %PYTHON_EXE%
    exit /b 1
)

echo [INFO] Force-reinstalling torch stack from cu128 index...
"%PYTHON_EXE%" -m pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
if errorlevel 1 (
    echo [ERROR] pip install failed.
    exit /b 1
)

echo [DONE] torch/torchvision/torchaudio force-reinstalled from cu128 index (torch pinned to 2.8.0).
exit /b 0

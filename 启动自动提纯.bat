@echo off
setlocal
chcp 65001 >nul
pushd "%~dp0" || exit /b 1
if not exist "python\python.exe" (
    echo Missing bundled python\python.exe
    pause
    exit /b 1
)
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONNOUSERSITE=1"
set "HF_HUB_OFFLINE=1"
set "PATH=%~dp0models\ffmpeg\bin;%PATH%"
"%~dp0python\python.exe" -X utf8 "%~dp0pipeline_app.py" %*
set "rc=%errorlevel%"
popd
if not "%rc%"=="0" pause
exit /b %rc%

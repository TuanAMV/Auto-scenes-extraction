@echo off
chcp 65001 >nul
echo [提示] 正在启动进程，请耐心等待...
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
"%~dp0python\python.exe" "%~dp0pipeline_app.py" %*
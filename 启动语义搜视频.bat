@echo off
chcp 65001 >nul 2>&1
echo 正在启动进程，请耐心等待...
echo.

:: 获取当前bat所在目录
set "SCRIPT_DIR=%~dp0"
:: 去掉末尾反斜杠
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

:: 嵌入式Python路径
set "PYTHON_EXE=%SCRIPT_DIR%\python\python.exe"

:: 检查Python是否存在
if not exist "%PYTHON_EXE%" (
    echo [错误] 未找到嵌入式Python: %PYTHON_EXE%
    pause
    exit /b 1
)

:: 设置项目根目录到PYTHONPATH
set "PYTHONPATH=%SCRIPT_DIR%;%PYTHONPATH%"

:: 启动 text_search.py
"%PYTHON_EXE%" "%SCRIPT_DIR%\text_search.py"

if %ERRORLEVEL% neq 0 (
    echo.
    echo [错误] 程序异常退出，错误码: %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause

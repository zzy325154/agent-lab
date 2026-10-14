@echo off
REM Agent 启动脚本。双击即可运行，或从命令行传参做单次提问。
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

if "%~1"=="" (
    python -m app.main
) else (
    python -m app.main %*
)
pause

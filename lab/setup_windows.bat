@echo off
rem Windows: one-time setup (creates lab\.venv and installs CUDA build of PyTorch)
rem Options: --cpu  --cuda cu130  --datasets
where py >nul 2>nul
if %errorlevel%==0 (
    py -3.13 "%~dp0setup.py" %*
) else (
    python "%~dp0setup.py" %*
)
pause

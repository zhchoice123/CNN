@echo off
rem Windows: start the training lab + study site, then open http://localhost:8766
cd /d "%~dp0.."
where py >nul 2>nul
if %errorlevel%==0 (
    py -3 lab\server.py
) else (
    python lab\server.py
)
pause

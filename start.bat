@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Starting microbial plate counter web service...
python webapp.py %*
if errorlevel 1 (
  echo.
  echo Failed to start. Try: pip install -r requirements.txt
  pause
)

@echo off
chcp 65001 > nul
cd /d "%~dp0"
title Hugging Face upload

if not exist ".venv\Scripts\python.exe" (
  echo Run 실행.bat once first.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" -c "from huggingface_hub import HfApi; HfApi().whoami()" > nul 2>&1
if not errorlevel 1 goto upload

echo.
echo ================================================================
echo  Hugging Face login (first time only)
echo  1. https://huggingface.co/settings/tokens  -  Create new token
echo  2. Token type: Write  -  Create  -  copy the token (hf_...)
echo  3. Paste it below with right-click, then press Enter
echo     (the token is not shown while pasting - that is normal)
echo  4. "Add token as git credential?"  -  type n, Enter
echo ================================================================
start "" "https://huggingface.co/settings/tokens"
".venv\Scripts\hf.exe" auth login
if errorlevel 1 goto fail

:upload
".venv\Scripts\python.exe" hf_deploy.py
if errorlevel 1 goto fail
pause
exit /b 0

:fail
echo.
echo Failed. Please send a screenshot of this window to Claude.
pause
exit /b 1

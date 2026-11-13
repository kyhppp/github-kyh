@echo off
chcp 65001 > nul
cd /d "%~dp0"
title 옛날 사진 복원기

if exist ".venv\Scripts\python.exe" goto run

echo [처음 실행] 프로그램 환경을 준비합니다. 몇 분 걸릴 수 있어요...
set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if exist "%PY%" goto mkvenv
set "PY=py"
:mkvenv
"%PY%" -m venv .venv
if errorlevel 1 goto nopython
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto fail

:run
echo 옛날 사진 복원기를 시작합니다. 잠시 후 브라우저가 열립니다.
echo 이 창을 닫으면 프로그램이 종료됩니다.
".venv\Scripts\python.exe" app.py
pause
exit /b 0

:nopython
echo Python 3.12가 필요합니다. 명령 프롬프트에서 다음을 실행한 뒤 다시 시도하세요:
echo   winget install Python.Python.3.12
:fail
pause
exit /b 1

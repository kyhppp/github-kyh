@echo off
chcp 65001 > nul
cd /d "%~dp0"
title 옛날 사진 복원기 (공개 링크)
if not exist ".venv\Scripts\python.exe" (
    echo 먼저 실행.bat을 한 번 실행해 환경을 준비하세요.
    pause
    exit /b 1
)
echo 공개 링크를 만드는 중입니다. 아래에 "Running on public URL: https://....gradio.live" 주소가 나타나면
echo 그 주소를 다른 사람에게 보내면 됩니다. 이 창을 닫으면 링크도 바로 끊깁니다.
echo.
".venv\Scripts\python.exe" app.py --share
pause

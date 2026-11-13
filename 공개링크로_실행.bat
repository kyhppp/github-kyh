@echo off
chcp 65001 > nul
cd /d "%~dp0"
title 옛날 사진 복원기 (공개 링크)

if not exist ".venv\Scripts\python.exe" goto novenv

rem ---- 이미 켜져 있으면 링크가 엉뚱한 곳을 가리키므로 중단
powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort 7860 -State Listen -ErrorAction SilentlyContinue) { exit 1 }"
if errorlevel 1 goto busy

rem ---- cloudflared 찾기 (없으면 한 번만 설치)
call :findcf
if defined CF goto havecf
echo [처음 한 번] 공개 링크 프로그램(cloudflared)을 설치합니다. 권한을 묻는 창이 뜨면 "예"를 눌러 주세요.
winget install --id Cloudflare.cloudflared -e --accept-source-agreements --accept-package-agreements
call :findcf
if not defined CF goto nocf

:havecf
set "LOG=%TEMP%\photo_restorer_tunnel.log"
set "URLFILE=%TEMP%\photo_restorer_url.txt"
if exist "%LOG%" del "%LOG%"
if exist "%URLFILE%" del "%URLFILE%"

echo 공개 링크를 만드는 중입니다... (10초쯤 걸려요)
start "" /b cmd /c ""%CF%" tunnel --no-autoupdate --url http://127.0.0.1:7860 > "%LOG%" 2>&1"

rem ---- 로그에서 주소를 찾아 파일에 적고 클립보드에 복사
powershell -NoProfile -Command "for ($i = 0; $i -lt 60; $i++) { if (Test-Path $env:LOG) { $m = Select-String -Path $env:LOG -Pattern 'https://[a-z0-9-]+\.trycloudflare\.com' | Select-Object -First 1; if ($m) { $u = $m.Matches[0].Value; Set-Content -Path $env:URLFILE -Value $u -Encoding ASCII; Set-Clipboard -Value $u; exit 0 } }; Start-Sleep 1 }; exit 1"
if errorlevel 1 goto nourl
set /p URL=<"%URLFILE%"

echo.
echo ================================================================
echo   공개 링크:  %URL%
echo   (클립보드에 복사해 뒀어요. 카톡·문자에 붙여넣기 하면 돼요)
echo ================================================================
echo   * 비밀번호 없이 링크만 있으면 누구나 쓸 수 있어요. 믿을 수 있는 사람에게만 보내세요.
echo   * 이 창을 닫으면 프로그램과 링크가 함께 꺼져요. 다시 켜면 링크 주소가 바뀌어요.
echo.
set GRADIO_SERVER_PORT=7860
".venv\Scripts\python.exe" app.py
pause
exit /b 0

:findcf
set "CF="
if exist "%ProgramFiles(x86)%\cloudflared\cloudflared.exe" set "CF=%ProgramFiles(x86)%\cloudflared\cloudflared.exe"
if not defined CF if exist "%ProgramFiles%\cloudflared\cloudflared.exe" set "CF=%ProgramFiles%\cloudflared\cloudflared.exe"
if not defined CF for /f "delims=" %%i in ('where cloudflared 2^>nul') do if not defined CF set "CF=%%i"
exit /b 0

:novenv
echo 먼저 실행.bat을 한 번 실행해 프로그램 환경을 준비하세요.
pause
exit /b 1

:busy
echo 옛날 사진 복원기가 이미 켜져 있어요. 먼저 켜 둔 창을 닫은 뒤 다시 실행하세요.
pause
exit /b 1

:nocf
echo cloudflared 설치에 실패했어요. 인터넷 연결을 확인하고 다시 실행해 주세요.
pause
exit /b 1

:nourl
echo 공개 링크를 만들지 못했어요. 인터넷 연결을 확인하고 다시 실행해 주세요.
echo 자세한 기록: %LOG%
pause
exit /b 1

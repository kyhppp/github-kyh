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
rem ---- Tailscale 고정 주소가 설정돼 있으면 함께 보여 준다 (README의 Tailscale 항목 참고)
set "TSURL="
set "TSFILE=%TEMP%\photo_restorer_ts.txt"
if exist "%TSFILE%" del "%TSFILE%"
powershell -NoProfile -Command "$ts = Join-Path $env:ProgramFiles 'Tailscale\tailscale.exe'; if (Test-Path $ts) { $s = & $ts serve status --json 2>$null | ConvertFrom-Json; $j = & $ts status --json 2>$null | ConvertFrom-Json; if ($j.BackendState -eq 'Running' -and $s.Web) { $h = @($s.Web.PSObject.Properties.Name)[0] -replace ':80$', ''; Set-Content -Path $env:TSFILE -Value ('http://' + $h) -Encoding ASCII } }" 2>nul
if exist "%TSFILE%" set /p TSURL=<"%TSFILE%"

echo 옛날 사진 복원기를 시작합니다. 잠시 후 브라우저가 열립니다.
echo   이 컴퓨터:  http://127.0.0.1:7860
if defined TSURL echo   다른 기기:  %TSURL%   (Tailscale 앱에 같은 계정으로 로그인한 기기에서만 열려요)
echo 이 창을 닫으면 프로그램이 종료됩니다.
set GRADIO_SERVER_PORT=7860
".venv\Scripts\python.exe" app.py
pause
exit /b 0

:nopython
echo Python 3.12가 필요합니다. 명령 프롬프트에서 다음을 실행한 뒤 다시 시도하세요:
echo   winget install Python.Python.3.12
:fail
pause
exit /b 1

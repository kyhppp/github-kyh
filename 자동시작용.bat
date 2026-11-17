@echo off
chcp 65001 > nul
rem 컴퓨터를 켤 때 복원기를 브라우저 없이 조용히 시작합니다 (시작프로그램용)
cd /d "%~dp0"
set NO_BROWSER=1
call "%~dp0실행.bat"

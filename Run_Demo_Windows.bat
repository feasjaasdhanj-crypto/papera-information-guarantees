@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
  py -3 Demo\run_all_demos.py
) else (
  where python >nul 2>nul
  if %ERRORLEVEL% EQU 0 (
    python Demo\run_all_demos.py
  ) else (
    echo Python 3 was not found. Install Python 3.13, then reopen this BAT.
    pause
    exit /b 1
  )
)
if errorlevel 1 (
  echo DEMO FAILED. See the error message above.
  pause
  exit /b 1
)
echo Demo PASS. Output saved to Demo\OUTPUT\
pause

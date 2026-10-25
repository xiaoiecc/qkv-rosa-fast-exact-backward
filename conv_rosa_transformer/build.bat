@echo off
rem Run in an x64 Visual Studio developer command prompt.
rem Usage: build.bat [your_script.py arguments...]
cd /d "%~dp0"
if defined CXX goto run
where cl >nul 2>nul
if errorlevel 1 (
  echo Open an x64 Native Tools Command Prompt for Visual Studio, then retry.
  exit /b 1
)
:run
if "%~1"=="" (
  python build.py
) else (
  python %*
)

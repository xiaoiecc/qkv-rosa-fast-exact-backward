@echo off
rem Build/run helper for the rosa C++ extension on Windows.
rem Usage: build.bat your_script.py [args...]
rem Adjust the vcvars64.bat path below to your Visual Studio / Build Tools install.
cd /d %~dp0
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
python %*

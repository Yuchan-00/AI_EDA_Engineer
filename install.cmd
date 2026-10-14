@echo off
rem AI EDA ENGINEER - Windows installer entry point: runs install.ps1 next to this file without changing the
rem execution policy of the machine (the -ExecutionPolicy Bypass applies to this one PowerShell process only).
rem Usage: install.cmd [-NoKicad] [-WithClaude] [-NoPath] [-SkipSelftest] [-Python 3.12]
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
exit /b %ERRORLEVEL%

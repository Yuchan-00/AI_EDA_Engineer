@echo off
rem AI EDA ENGINEER - Windows installer entry point: runs install.ps1 next to this file in Windows PowerShell 5.1
rem without changing the execution policy of the machine (-ExecutionPolicy Bypass applies to this one process only).
rem Usage: install.cmd [-Dir DIR] [-Branch NAME] [-Python VER] [-NoKicad] [-WithClaude] [-NoPath] [-SkipSelftest]
rem (each option also has an environment variable: see the header of install.ps1)
rem A 32-bit cmd.exe starts the 64-bit PowerShell through Sysnative; PowerShell is found by its own path, not PATH.
setlocal
set "WINPS=%SystemRoot%\Sysnative\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%WINPS%" set "WINPS=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%WINPS%" set "WINPS=powershell.exe"
"%WINPS%" -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
exit /b %ERRORLEVEL%

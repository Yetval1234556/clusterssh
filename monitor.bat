@echo off
setlocal enabledelayedexpansion

:: ── Config — update these ──────────────────────────────────────────────────────
set UNC_HOST=longleaf.unc.edu
set UNC_USER=rpatel1
set ORACLE_HOST=YOUR_ORACLE_INSTANCE_IP
set ORACLE_USER=opc
set REFRESH=10
:: ──────────────────────────────────────────────────────────────────────────────

:: Usage:
::   monitor.bat unc              - monitor UNC H200 cluster
::   monitor.bat oracle           - monitor Oracle A100 instance
::   monitor.bat unc connect      - just SSH into UNC
::   monitor.bat oracle connect   - just SSH into Oracle

set TARGET=%~1
if "%TARGET%"=="" set TARGET=unc

set MODE=monitor
if /i "%~2"=="connect" set MODE=connect

if /i "%TARGET%"=="oracle" (
    set HOST=%ORACLE_HOST%
    set SSH_USER=%ORACLE_USER%
    set SCRATCH=$HOME/bloomi
) else (
    set HOST=%UNC_HOST%
    set SSH_USER=%UNC_USER%
    set SCRATCH=/scratch/rpatel1/bloomi
)

set SSH_OPTS=-o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ConnectTimeout=10
set SCRIPTDIR=%~dp0

:: ── Connect mode ───────────────────────────────────────────────────────────────
if "%MODE%"=="connect" (
    echo Connecting to %SSH_USER%@%HOST%...
    :connect_loop
    ssh %SSH_OPTS% %SSH_USER%@%HOST%
    if errorlevel 1 (
        echo Connection lost. Reconnecting in 5s... (Ctrl+C to stop)
        timeout /t 5 /nokey >nul
        goto connect_loop
    )
    exit /b 0
)

:: ── Monitor mode ───────────────────────────────────────────────────────────────
echo ========================================================
echo   DinoBloom-G Training Monitor
echo   Target  : %TARGET% (%SSH_USER%@%HOST%)
echo   Refresh : every %REFRESH%s
echo   Ctrl+C  to stop
echo ========================================================
echo.

:monitor_loop
cls
echo ========================================================
echo   DinoBloom-G -- %date% %time%
echo   %TARGET% ^| %SSH_USER%@%HOST%
echo ========================================================

ssh %SSH_OPTS% %SSH_USER%@%HOST% "export SCRATCH=%SCRATCH%; bash -s" < "%SCRIPTDIR%_monitor_remote.sh"

if errorlevel 1 (
    echo.
    echo Connection lost. Reconnecting in 5s... (Ctrl+C to stop)
    timeout /t 5 /nokey >nul
    goto monitor_loop
)

echo.
echo Refreshing in %REFRESH%s... (Ctrl+C to stop)
timeout /t %REFRESH% /nokey >nul
goto monitor_loop

@echo off
setlocal enabledelayedexpansion

:: ── Config ─────────────────────────────────────────────────────────────────────
set CLUSTER_HOST=login-01.ncshare.org
set CLUSTER_USER=rpatel1
set SCRATCH=/hpc/home/rpatel1/bloomi
set REFRESH=10
:: ──────────────────────────────────────────────────────────────────────────────

:: Usage:
::   .\monitor.bat             - live dashboard
::   .\monitor.bat connect     - plain SSH into cluster

set MODE=monitor
if /i "%~1"=="connect" set MODE=connect

set SSH_OPTS=-o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ConnectTimeout=10
set SCRIPTDIR=%~dp0

:: ── Connect mode ───────────────────────────────────────────────────────────────
if "%MODE%"=="connect" (
    echo Connecting to %CLUSTER_USER%@%CLUSTER_HOST%...
    :connect_loop
    ssh %SSH_OPTS% %CLUSTER_USER%@%CLUSTER_HOST%
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
echo   Host    : %CLUSTER_USER%@%CLUSTER_HOST%
echo   Refresh : every %REFRESH%s
echo   Ctrl+C  to stop
echo ========================================================
echo.

:monitor_loop
cls
echo ========================================================
echo   DinoBloom-G -- %date% %time%
echo   %CLUSTER_USER%@%CLUSTER_HOST%
echo ========================================================

ssh %SSH_OPTS% %CLUSTER_USER%@%CLUSTER_HOST% "export SCRATCH=%SCRATCH%; bash -s" < "%SCRIPTDIR%_monitor_remote.sh"

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

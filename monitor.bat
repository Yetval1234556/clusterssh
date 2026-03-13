@echo off
setlocal enabledelayedexpansion

:: ── Config ──────────────────────────────────────────────────────────────────
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

set SSH=ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ConnectTimeout=10 %CLUSTER_USER%@%CLUSTER_HOST%

:: ── Connect mode ────────────────────────────────────────────────────────────
if "%MODE%"=="connect" (
    echo Connecting to %CLUSTER_USER%@%CLUSTER_HOST%...
    :connect_loop
    %SSH%
    if errorlevel 1 (
        echo Connection lost. Reconnecting in 5s... (Ctrl+C to stop)
        timeout /t 5 /nokey >nul
        goto connect_loop
    )
    exit /b 0
)

:: ── Monitor mode ─────────────────────────────────────────────────────────────
echo ========================================================
echo   DinoBloom-G Training Monitor
echo   Host    : %CLUSTER_USER%@%CLUSTER_HOST%
echo   Refresh : every %REFRESH%s  ^|  Ctrl+C to stop
echo ========================================================
echo.

:monitor_loop
cls
echo ========================================================
echo   DinoBloom-G -- %date% %time%
echo   %CLUSTER_USER%@%CLUSTER_HOST%
echo ========================================================

%SSH% "echo '' && echo '--- GPU Status ---' && nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.free,temperature.gpu,power.draw --format=csv,noheader 2>/dev/null && echo '' && echo '--- SLURM Jobs ---' && squeue -u $USER && echo '' && echo '--- Training Metrics ---' && if [ -f %SCRATCH%/training_metrics.csv ]; then tail -6 %SCRATCH%/training_metrics.csv; else echo 'No metrics yet.'; fi && echo '' && echo '--- Recent Log (last 20 lines) ---' && LATEST=$(ls -t %SCRATCH%/logs/dino_*.out 2>/dev/null | head -1) && if [ -n \"$LATEST\" ]; then echo \"  $LATEST\" && tail -20 \"$LATEST\"; else echo 'No log file yet.'; fi"

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

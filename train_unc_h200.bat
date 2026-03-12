@echo off
setlocal

:: ── Config — update these ──────────────────────────────────────────────────────
set UNC_HOST=ncshare.login.org
set UNC_USER=rpatel1
:: ──────────────────────────────────────────────────────────────────────────────

:: Usage:
::   train_unc_h200.bat      (8 GPUs default)
::   train_unc_h200.bat 1    (1 GPU)

set NGPUS=%~1
if "%NGPUS%"=="" set NGPUS=8

set SCRIPTDIR=%~dp0

echo Copying train script to cluster...
scp "%SCRIPTDIR%train_unc_h200.sh" %UNC_USER%@%UNC_HOST%:~/train_unc_h200.sh
if errorlevel 1 (
    echo ERROR: SCP failed. Check your host and username in train_unc_h200.bat.
    exit /b 1
)

echo Submitting SLURM job with %NGPUS% GPU(s)...
ssh -o ConnectTimeout=10 %UNC_USER%@%UNC_HOST% "sbatch ~/train_unc_h200.sh %NGPUS%"

echo.
echo Monitor with:  monitor.bat unc
echo View queue  :  ssh %UNC_USER%@%UNC_HOST% squeue -u %UNC_USER%

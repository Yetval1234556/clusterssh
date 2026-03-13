@echo off
setlocal

:: ── Config — update these ──────────────────────────────────────────────────────
set UNC_HOST=login-01.ncshare.org
set UNC_USER=rpatel1
:: ──────────────────────────────────────────────────────────────────────────────

:: Usage:
::   train_h200.bat      (4 GPUs default)
::   train_h200.bat 2    (2 GPUs)
::   train_h200.bat 1    (1 GPU)

set NGPUS=%~1
if "%NGPUS%"=="" set NGPUS=4

set SCRIPTDIR=%~dp0

echo Copying launcher scripts to cluster...
scp "%SCRIPTDIR%train_h200.sh" "%SCRIPTDIR%epoch_report.py" %UNC_USER%@%UNC_HOST%:~/
if errorlevel 1 (
    echo ERROR: SCP failed. Check your host and username in train_h200.bat.
    exit /b 1
)

echo Copying training scripts to cluster...
scp "%SCRIPTDIR%train_efficientnet_b0.py" "%SCRIPTDIR%train_efficientnet_b0_ddp.py" "%SCRIPTDIR%conda.yaml" "%SCRIPTDIR%epoch_report.py" %UNC_USER%@%UNC_HOST%:~/bloomi/
if errorlevel 1 (
    echo ERROR: Failed to copy training scripts.
    exit /b 1
)

echo Submitting SLURM job with %NGPUS% GPU(s)...
ssh -o ConnectTimeout=10 %UNC_USER%@%UNC_HOST% "sbatch ~/train_h200.sh %NGPUS%"

echo.
echo Monitor with:  monitor.bat
echo View queue  :  ssh %UNC_USER%@%UNC_HOST% squeue -u %UNC_USER%

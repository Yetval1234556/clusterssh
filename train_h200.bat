@echo off
setlocal enabledelayedexpansion

:: ── Config ──────────────────────────────────────────────────────────────────
set UNC_HOST=login-01.ncshare.org
set UNC_USER=rpatel1
:: ──────────────────────────────────────────────────────────────────────────────

:: Usage:
::   train_h200.bat      (4 GPUs default)
::   train_h200.bat 2    (2 GPUs)
::   train_h200.bat 1    (1 GPU)

set NGPUS=%~1
if "%NGPUS%"=="" set NGPUS=1

set SCRIPTDIR=%~dp0
set SSH=ssh -o ConnectTimeout=10 %UNC_USER%@%UNC_HOST%

echo.
echo ========================================================
echo   DinoBloom-G Training Launcher
echo   Target  : %UNC_USER%@%UNC_HOST%
echo   GPUs    : %NGPUS%x H200
echo   Epochs  : 75   LR: 1e-4   Batch/GPU: 64
echo   Effective batch: %NGPUS% x 64
echo ========================================================
echo.

:: ── [1] Launcher scripts ────────────────────────────────────────────────────
echo [1/5] Copying launcher scripts to cluster...
echo   ^> train_h200.sh
echo   ^> epoch_report.py
scp "%SCRIPTDIR%train_h200.sh" "%SCRIPTDIR%epoch_report.py" %UNC_USER%@%UNC_HOST%:~/
if errorlevel 1 (
    echo.
    echo   ERROR: SCP failed. Check SSH access and .bat config.
    exit /b 1
)
echo   Done.
echo.

:: ── [2] Training scripts ────────────────────────────────────────────────────
echo [2/5] Copying training scripts to cluster...
echo   ^> train_efficientnet_b0.py
echo   ^> train_efficientnet_b0_ddp.py
echo   ^> requirements.txt
echo   ^> epoch_report.py
scp "%SCRIPTDIR%train_efficientnet_b0.py" "%SCRIPTDIR%train_efficientnet_b0_ddp.py" "%SCRIPTDIR%requirements.txt" "%SCRIPTDIR%epoch_report.py" %UNC_USER%@%UNC_HOST%:~/bloomi/
if errorlevel 1 (
    echo.
    echo   ERROR: Failed to copy training scripts.
    exit /b 1
)
echo   Done.
echo.

:: ── [3] dinov2 package ──────────────────────────────────────────────────────
echo [3/5] Checking dinov2 package on cluster...
for /f %%i in ('%SSH% "test -d ~/bloomi/dinov2 && echo yes || echo no"') do set DINOV2=%%i
if "%DINOV2%"=="yes" (
    echo   dinov2/ already on cluster — skipping ^(saves time^).
) else (
    echo   Not found — copying dinov2/ ^(first time only^)...
    scp -r "%SCRIPTDIR%dinov2" %UNC_USER%@%UNC_HOST%:~/bloomi/
    if errorlevel 1 (
        echo   ERROR: Failed to copy dinov2 package.
        exit /b 1
    )
    echo   Done.
)
echo.

:: ── [4] train/val split files ───────────────────────────────────────────────
echo [4/5] Checking train/val split files on cluster...
for /f %%i in ('%SSH% "test -f ~/bloomi/New\ Data/train.txt && echo yes || echo no"') do set TXTS=%%i
if "%TXTS%"=="yes" (
    echo   train.txt / val.txt already on cluster — skipping.
) else (
    echo   Not found — copying train.txt + val.txt...
    %SSH% "mkdir -p ~/bloomi/New\ Data"
    scp "%SCRIPTDIR%New Data\train.txt" "%SCRIPTDIR%New Data\val.txt" "%UNC_USER%@%UNC_HOST%:~/bloomi/New Data/"
    if errorlevel 1 (
        echo   ERROR: Failed to copy train.txt / val.txt.
        exit /b 1
    )
    echo   Done.
)
echo.

:: ── [5] Submit sbatch job ───────────────────────────────────────────────────
echo [5/5] Submitting SLURM job (%NGPUS% GPU(s))...
for /f "tokens=*" %%j in ('%SSH% "sbatch --gres=gpu:h200:%NGPUS% ~/train_h200.sh %NGPUS%"') do (
    set SBATCH_OUT=%%j
    echo   %%j
)
echo.

echo ========================================================
echo   Job submitted successfully!
echo.
echo   GPUs    : %NGPUS%x H200 ^(96GB VRAM each^)
echo   Script  : ~/train_h200.sh
echo   Log dir : ~/bloomi/logs/
echo.
echo   Monitor commands:
echo     monitor.bat                          ^(live dashboard^)
echo     monitor.bat connect                  ^(SSH shell^)
echo     ssh %UNC_USER%@%UNC_HOST% squeue -u %UNC_USER%
echo     ssh %UNC_USER%@%UNC_HOST% "tail -f ~/bloomi/logs/dino_*.out"
echo ========================================================
echo.

@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: ── Config ────────────────────────────────────────────────────────────────────
set CLUSTER_HOST=login-01.ncshare.org
set CLUSTER_USER=rpatel1
:: ──────────────────────────────────────────────────────────────────────────────

set NGPUS=%~1
if "%NGPUS%"=="" set NGPUS=1

set SCRIPTDIR=%~dp0
set SSH=ssh %CLUSTER_USER%@%CLUSTER_HOST%
set SCP=scp

echo.
echo  ============================================================
echo    BLOOM -- Training Launcher
echo    Host   : %CLUSTER_USER%@%CLUSTER_HOST%
echo    GPUs   : %NGPUS%x H200  (96GB VRAM)
echo    Epochs : 100  (adaptive batch + workers based on dataset)
echo  ============================================================
echo.

:: ── [0a] Git pull ─────────────────────────────────────────────────────────────
echo  [0a] Pulling latest from GitHub...
pushd "%SCRIPTDIR%"
git remote get-url origin >nul 2>&1
if errorlevel 1 git remote add origin https://github.com/Yetval1234556/clusterssh.git
git pull origin master
if errorlevel 1 (
    echo    WARNING: git pull failed - using local files.
) else (
    echo    OK: up to date.
)
popd
echo.

:: ── [0b] SSH key ──────────────────────────────────────────────────────────────
echo  [0b] Loading SSH key...
sc start ssh-agent >nul 2>&1
ssh-add %USERPROFILE%\.ssh\id_ed25519
echo.

:: ── [1] Launcher + training scripts ──────────────────────────────────────────
echo  [1] Copying scripts to cluster...
%SCP% "%SCRIPTDIR%train_h200.sh" "%SCRIPTDIR%epoch_report.py" %CLUSTER_USER%@%CLUSTER_HOST%:~/
if errorlevel 1 ( echo    ERROR: Failed to copy launcher scripts. & exit /b 1 )

%SCP% "%SCRIPTDIR%train_efficientnet_b0.py" "%SCRIPTDIR%train_efficientnet_b0_ddp.py" "%SCRIPTDIR%requirements.txt" "%SCRIPTDIR%epoch_report.py" "%SCRIPTDIR%make_splits.py" %CLUSTER_USER%@%CLUSTER_HOST%:~/bloomi/
if errorlevel 1 ( echo    ERROR: Failed to copy training scripts. & exit /b 1 )
echo    OK: Scripts copied.
echo.

:: ── [2] dinov2 package ────────────────────────────────────────────────────────
echo  [2] Checking dinov2 package on cluster...
for /f %%i in ('%SSH% "test -d ~/bloomi/dinov2 && echo yes || echo no"') do set DINOV2=%%i
if "%DINOV2%"=="yes" (
    echo    SKIP: dinov2 already on cluster.
) else (
    echo    Uploading dinov2 package...
    %SCP% -r "%SCRIPTDIR%dinov2" %CLUSTER_USER%@%CLUSTER_HOST%:~/bloomi/
    if errorlevel 1 ( echo    ERROR: Failed to copy dinov2. & exit /b 1 )
    echo    OK: dinov2 uploaded.
)
echo.

:: ── [3] Submit job ────────────────────────────────────────────────────────────
echo  [3] Submitting SLURM job...
echo    Fixing line endings on train_h200.sh...
%SSH% "sed -i 's/\r//' ~/train_h200.sh"

for /f "tokens=*" %%j in ('%SSH% "sbatch --gres=gpu:h200:%NGPUS% ~/train_h200.sh %NGPUS%"') do (
    set SBATCH_OUT=%%j
    echo    %%j
)
echo.

:: Extract job ID
for /f "tokens=4" %%j in ("!SBATCH_OUT!") do set JOBID=%%j

echo  ============================================================
echo    Job submitted!
echo.
echo    Monitor queue  : ssh %CLUSTER_USER%@%CLUSTER_HOST% "squeue -u %CLUSTER_USER%"
echo    Watch logs     : ssh %CLUSTER_USER%@%CLUSTER_HOST% "tail -f ~/bloomi/logs/dino_!JOBID!.out"
echo    Cancel job     : ssh %CLUSTER_USER%@%CLUSTER_HOST% "scancel !JOBID!"
echo  ============================================================
echo.

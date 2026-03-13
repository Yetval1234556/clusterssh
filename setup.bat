@echo off
setlocal enabledelayedexpansion

:: ── Config ────────────────────────────────────────────────────────────────────
set CLUSTER_HOST=login-01.ncshare.org
set CLUSTER_USER=rpatel1
set OCI_KEY=%~dp0yetvald@gmail.com-2026-03-13T02_48_12.614Z.pem
:: ──────────────────────────────────────────────────────────────────────────────

set SCRIPTDIR=%~dp0
set SSH=ssh -o ConnectTimeout=10 %CLUSTER_USER%@%CLUSTER_HOST%

:: Pull latest from GitHub before anything else
echo [0a] Pulling latest from GitHub...
git -C "%SCRIPTDIR%" pull
if errorlevel 1 ( echo WARNING: git pull failed — using local files. )
echo.

:: Cache SSH passphrase once for the whole session
echo [0] Caching SSH key passphrase (enter once, reused for all steps)...
sc start ssh-agent >nul 2>&1
ssh-add %USERPROFILE%\.ssh\id_ed25519 2>nul
echo.

:: Always copy launcher scripts (tiny, always want latest)
echo [1] Copying launcher scripts...
scp "%SCRIPTDIR%setup.sh" "%SCRIPTDIR%train_h200.sh" "%SCRIPTDIR%epoch_report.py" %CLUSTER_USER%@%CLUSTER_HOST%:~/
if errorlevel 1 ( echo ERROR: SCP failed. & exit /b 1 )

:: Always copy training scripts (fast, always want latest)
echo [2] Copying training scripts...
%SSH% "mkdir -p ~/bloomi"
scp "%SCRIPTDIR%train_efficientnet_b0.py" "%SCRIPTDIR%train_efficientnet_b0_ddp.py" "%SCRIPTDIR%requirements.txt" "%SCRIPTDIR%epoch_report.py" %CLUSTER_USER%@%CLUSTER_HOST%:~/bloomi/
if errorlevel 1 ( echo ERROR: Failed to copy training scripts. & exit /b 1 )

:: Skip dinov2 if already on cluster
echo [3] Checking dinov2 package...
for /f %%i in ('%SSH% "test -d ~/bloomi/dinov2 && echo yes || echo no"') do set DINOV2=%%i
if "%DINOV2%"=="yes" (
    echo     dinov2 already on cluster — skipping.
) else (
    echo     Copying dinov2 package ^(first time only^)...
    scp -r "%SCRIPTDIR%dinov2" %CLUSTER_USER%@%CLUSTER_HOST%:~/bloomi/
    if errorlevel 1 ( echo ERROR: Failed to copy dinov2. & exit /b 1 )
)

:: Skip train/val txt if already on cluster
echo [4] Checking train/val split files...
for /f %%i in ('%SSH% "test -f ~/bloomi/New\ Data/train.txt && echo yes || echo no"') do set TXTS=%%i
if "%TXTS%"=="yes" (
    echo     train.txt / val.txt already on cluster — skipping.
) else (
    echo     Copying train/val split files...
    %SSH% "mkdir -p ~/bloomi/New\ Data"
    scp "%SCRIPTDIR%New Data\train.txt" "%SCRIPTDIR%New Data\val.txt" "%CLUSTER_USER%@%CLUSTER_HOST%:~/bloomi/New Data/"
    if errorlevel 1 ( echo ERROR: Failed to copy txt files. & exit /b 1 )
)

:: Skip OCI key if already on cluster
echo [5] Checking OCI key...
for /f %%i in ('%SSH% "test -f ~/.oci/oci_api_key.pem && echo yes || echo no"') do set OCIKEY=%%i
if "%OCIKEY%"=="yes" (
    echo     OCI key already on cluster — skipping.
) else (
    echo     Copying OCI private key...
    %SSH% "mkdir -p ~/.oci && chmod 700 ~/.oci"
    scp "%OCI_KEY%" %CLUSTER_USER%@%CLUSTER_HOST%:~/.oci/oci_api_key.pem
    if errorlevel 1 (
        echo ERROR: Failed to copy OCI key. Check it exists at: %OCI_KEY%
        exit /b 1
    )
    %SSH% "chmod 600 ~/.oci/oci_api_key.pem"
    echo     OCI key copied and secured.
)

:: Force re-download of DinoBloom-G weights from Oracle (always use latest)
echo [6] Clearing old DinoBloom-G weights on cluster (will re-download fresh from Oracle)...
%SSH% "rm -f ~/bloomi/DinoBloom-G.pth && echo     Cleared."

:: Run setup on cluster
echo [7] Running setup on cluster...
%SSH% "sed -i 's/\r//' ~/setup.sh && bash ~/setup.sh"

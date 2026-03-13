@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: ═══════════════════════════════════════════════════════════════════════════════
::   BLOOM — Oracle Cluster Setup  (Smart Edition)
::   Syncs code, data, and keys to login-01.ncshare.org
::
::   Smart features:
::     - Pre-flight SSH test + disk space check before touching anything
::     - Dataset synced from Google Drive folder (gdown --remaining-ok, skips existing)
::     - Shows before/after archive inventory with sizes
::     - DinoBloom-G weights pulled from Google Drive (gdown, skips if already present)
::     - Skips every step already done on the cluster
::     - Auto-generates train.txt / val.txt via make_splits.py if missing
:: ═══════════════════════════════════════════════════════════════════════════════

:: ── Configuration ─────────────────────────────────────────────────────────────
set CLUSTER_HOST=login-01.ncshare.org
set CLUSTER_USER=rpatel1
set OCI_KEY=%~dp0yetvald@gmail.com-2026-03-13T02_52_21.874Z.pem
set GDRIVE_ID=1J5ld-tK6cewj9wXWUi3rs6UdlHnDBe8U
set GDRIVE_URL=https://drive.google.com/drive/folders/1J5ld-tK6cewj9wXWUi3rs6UdlHnDBe8U
:: ──────────────────────────────────────────────────────────────────────────────

:: Strip trailing backslash from SCRIPTDIR (prevents git -C quoting issues)
set SCRIPTDIR=%~dp0
if "%SCRIPTDIR:~-1%"=="\" set SCRIPTDIR=%SCRIPTDIR:~0,-1%
set SSH=ssh -o ConnectTimeout=15 %CLUSTER_USER%@%CLUSTER_HOST%

echo.
echo  ╔═══════════════════════════════════════════════════════════════════════╗
echo  ║          B L O O M  —  Oracle Cluster Setup                          ║
echo  ║          Host   : login-01.ncshare.org                               ║
echo  ║          User   : rpatel1                                            ║
echo  ║          Drive  : Google Drive (public folder)                       ║
echo  ╚═══════════════════════════════════════════════════════════════════════╝
echo.

:: ── [0a] Git Pull ─────────────────────────────────────────────────────────────
echo  ┌───────────────────────────────────────────────────────────────────────┐
echo  │  [0a]  Pull latest from GitHub                                        │
echo  └───────────────────────────────────────────────────────────────────────┘
echo.
git -C "%SCRIPTDIR%" pull origin master
if errorlevel 1 (
    echo    WARNING: git pull failed — using local files.
) else (
    echo    OK: Repository is up to date.
)
echo.

:: ── [0b] SSH Key Cache — must happen BEFORE any SSH/SCP calls ────────────────
echo  ┌───────────────────────────────────────────────────────────────────────┐
echo  │  [0b]  Cache SSH key  (enter passphrase once, reused for all steps)   │
echo  └───────────────────────────────────────────────────────────────────────┘
echo.
:: Try to start the Windows OpenSSH Authentication Agent service
net start ssh-agent >nul 2>&1
sc start ssh-agent >nul 2>&1

:: Check if agent is now reachable
ssh-add -l >nul 2>&1
if errorlevel 1 (
    echo    NOTE: ssh-agent service is not running.
    echo    To fix permanently, run once in PowerShell as Administrator:
    echo      Set-Service ssh-agent -StartupType Automatic
    echo      Start-Service ssh-agent
    echo    Continuing without agent — you may be prompted for passphrase each step.
    echo.
)
ssh-add %USERPROFILE%\.ssh\id_ed25519
echo    OK: SSH key loaded.
echo.

:: ── Pre-flight: test SSH connectivity (after key is loaded) ───────────────────
echo  [preflight] Testing SSH connection to cluster...
%SSH% "echo OK" >nul 2>&1
if errorlevel 1 (
    echo.
    echo  ERROR: Cannot reach %CLUSTER_HOST%
    echo.
    echo  Most common causes:
    echo    1. Not on VPN — connect to UNC VPN first
    echo    2. SSH key not found at %USERPROFILE%\.ssh\id_ed25519
    echo    3. Wrong passphrase entered above
    echo.
    echo  Debug with:  ssh -v rpatel1@login-01.ncshare.org
    exit /b 1
)
echo  [preflight] SSH OK
echo.

:: ── Pre-flight: cluster disk space ────────────────────────────────────────────
echo  [preflight] Cluster disk usage:
%SSH% "df -h ~ | tail -1 | awk '{printf \"    Home: used=%%s  avail=%%s  (%%s full)\n\", $3, $4, $5}'"
echo.

:: ── Pre-flight: what archives are already on cluster ──────────────────────────
echo  [preflight] Scanning cluster for existing archives...
echo.
%SSH% "mkdir -p ~/bloomi/'New Data'/extracted && echo '    Archive inventory:' && ls ~/bloomi/'New Data'/extracted/ 2>/dev/null | grep -E '^archive[0-9]+' | sort -V | while read a; do echo \"      $a\"; done || echo '      (none yet)'"
echo.

:: ── [1] Launcher Scripts ──────────────────────────────────────────────────────
echo  ┌───────────────────────────────────────────────────────────────────────┐
echo  │  [1]   Launcher scripts  →  ~/                                        │
echo  └───────────────────────────────────────────────────────────────────────┘
echo.
echo    Files  : setup.sh, train_h200.sh, epoch_report.py
echo    Target : %CLUSTER_USER%@%CLUSTER_HOST%:~/
echo.
scp "%SCRIPTDIR%setup.sh" "%SCRIPTDIR%train_h200.sh" "%SCRIPTDIR%epoch_report.py" %CLUSTER_USER%@%CLUSTER_HOST%:~/
if errorlevel 1 ( echo    ERROR: SCP failed — aborting. & exit /b 1 )
echo    OK: Launcher scripts copied.
echo.

:: ── [2] Training Scripts ──────────────────────────────────────────────────────
echo  ┌───────────────────────────────────────────────────────────────────────┐
echo  │  [2]   Training scripts  →  ~/bloomi/                                 │
echo  └───────────────────────────────────────────────────────────────────────┘
echo.
echo    Files  : train_efficientnet_b0.py, train_efficientnet_b0_ddp.py
echo             requirements.txt, epoch_report.py
echo    Target : %CLUSTER_USER%@%CLUSTER_HOST%:~/bloomi/
echo.
%SSH% "mkdir -p ~/bloomi"
scp "%SCRIPTDIR%train_efficientnet_b0.py" "%SCRIPTDIR%train_efficientnet_b0_ddp.py" "%SCRIPTDIR%requirements.txt" "%SCRIPTDIR%epoch_report.py" "%SCRIPTDIR%make_splits.py" %CLUSTER_USER%@%CLUSTER_HOST%:~/bloomi/
if errorlevel 1 ( echo    ERROR: Failed to copy training scripts — aborting. & exit /b 1 )
echo    OK: Training scripts copied.
echo.

:: ── [3] DinoV2 Package ────────────────────────────────────────────────────────
echo  ┌───────────────────────────────────────────────────────────────────────┐
echo  │  [3]   dinov2 package  (skip if already on cluster)                   │
echo  └───────────────────────────────────────────────────────────────────────┘
echo.
for /f %%i in ('%SSH% "test -d ~/bloomi/dinov2 && echo yes || echo no"') do set DINOV2=%%i
if "%DINOV2%"=="yes" (
    echo    SKIP: dinov2 already at ~/bloomi/dinov2/
) else (
    echo    NOT FOUND — uploading dinov2 package ^(first time only^)...
    scp -r "%SCRIPTDIR%dinov2" %CLUSTER_USER%@%CLUSTER_HOST%:~/bloomi/
    if errorlevel 1 ( echo    ERROR: Failed to copy dinov2 — aborting. & exit /b 1 )
    echo    OK: dinov2 uploaded.
)
echo.

:: ── [4] Dataset: Google Drive folder → Oracle cluster (smart sync) ───────────
echo  ┌───────────────────────────────────────────────────────────────────────┐
echo  │  [4]   Dataset archives: Google Drive  →  Oracle cluster             │
echo  │        (gdown --remaining-ok skips files already present)            │
echo  └───────────────────────────────────────────────────────────────────────┘
echo.
echo    Google Drive : %GDRIVE_URL%
echo    Cluster dest : ~/bloomi/New Data/extracted/
echo.

:: Check disk space first
echo    Disk space on cluster:
%SSH% "df -h ~ | tail -1 | awk '{printf \"    used=%%s  avail=%%s  (%%s full)\n\", $3, $4, $5}'"
echo.

:: Count archives before sync
for /f %%C in ('%SSH% "ls ~/bloomi/'New Data'/extracted/ 2>/dev/null | grep -cE '^archive[0-9]+' || echo 0"') do set BEFORE_COUNT=%%C
echo    Archives on cluster BEFORE sync: %BEFORE_COUNT%
echo.

:: Ensure gdown is installed on cluster
echo    Ensuring gdown is installed on cluster...
%SSH% "pip show gdown >nul 2>&1 && echo '    gdown already installed.' || (pip install -q gdown && echo '    gdown installed successfully.')"
echo.

:: Smart sync — gdown skips files already present
echo    Starting smart sync from Google Drive...
echo    (gdown will skip archives already present, download only new ones)
echo.
%SSH% "cd ~/bloomi/'New Data'/extracted && gdown --folder https://drive.google.com/drive/folders/%GDRIVE_ID% --remaining-ok && echo SYNC_COMPLETE"
if errorlevel 1 (
    echo.
    echo    ERROR: gdown sync failed — aborting.
    echo    Check that the folder is set to public ^(anyone with link^).
    exit /b 1
)
echo.

:: Show inventory after sync
for /f %%C in ('%SSH% "ls ~/bloomi/'New Data'/extracted/ 2>/dev/null | grep -cE '^archive[0-9]+' || echo 0"') do set AFTER_COUNT=%%C
set /a NEW_ARCHIVES=%AFTER_COUNT% - %BEFORE_COUNT%
echo    ┌────────────────────────────────────────────────────────────────────┐
echo    │   Archive inventory AFTER sync                                     │
echo    ├────────────────────────────────────────────────────────────────────┤
%SSH% "ls ~/bloomi/'New Data'/extracted/ 2>/dev/null | grep -E '^archive[0-9]+' | sort -V | while read a; do SIZE=$(du -sh ~/bloomi/'New Data'/extracted/$a 2>/dev/null | cut -f1); echo \"    │   $a   ($SIZE)\"; done || echo '    │   (none found)'"
echo    ├────────────────────────────────────────────────────────────────────┤
echo    │   Before: %BEFORE_COUNT% archives   After: %AFTER_COUNT% archives   New: %NEW_ARCHIVES%
echo    └────────────────────────────────────────────────────────────────────┘
echo.
echo    OK: Dataset sync complete.
echo.

:: ── [5] Image Count + Train/Val Split ────────────────────────────────────────
echo  ┌───────────────────────────────────────────────────────────────────────┐
echo  │  [5]   Count images  +  generate train.txt / val.txt if missing       │
echo  │        (runs make_splits.py on the cluster — stratified 80/20 split)  │
echo  └───────────────────────────────────────────────────────────────────────┘
echo.
echo    Running make_splits.py on cluster...
echo    - Shows total image count per archive and per class
echo    - Skips split generation if train.txt / val.txt already exist
echo    - Auto-generates a stratified 80/20 split if they are missing
echo.
%SSH% "python3 ~/bloomi/make_splits.py"
if errorlevel 1 (
    echo    ERROR: make_splits.py failed — aborting.
    echo    Check that archives downloaded correctly in step [4].
    exit /b 1
)
echo.

:: ── [6] OCI Private Key ───────────────────────────────────────────────────────
echo  ┌───────────────────────────────────────────────────────────────────────┐
echo  │  [6]   OCI private key  (skip if already on cluster)                  │
echo  └───────────────────────────────────────────────────────────────────────┘
echo.
for /f %%i in ('%SSH% "test -f ~/.oci/oci_api_key.pem && echo yes || echo no"') do set OCIKEY=%%i
if "%OCIKEY%"=="yes" (
    echo    SKIP: OCI key already at ~/.oci/oci_api_key.pem
) else (
    echo    NOT FOUND — uploading OCI key...
    echo    Local key : %OCI_KEY%
    %SSH% "mkdir -p ~/.oci && chmod 700 ~/.oci"
    scp "%OCI_KEY%" %CLUSTER_USER%@%CLUSTER_HOST%:~/.oci/oci_api_key.pem
    if errorlevel 1 (
        echo    ERROR: Failed to copy OCI key — aborting.
        echo    Expected : %OCI_KEY%
        exit /b 1
    )
    %SSH% "chmod 600 ~/.oci/oci_api_key.pem"
    echo    OK: OCI key uploaded and secured ^(chmod 600^).
)
echo.

:: ── [7] Clear original DinoBloom-G model ────────────────────────────────────
echo  ┌───────────────────────────────────────────────────────────────────────┐
echo  │  [7]   Clear original DinoBloom-G model  (setup.sh re-downloads it   │
echo  │        from Google Drive — this is the base we fine-tune on top of)  │
echo  └───────────────────────────────────────────────────────────────────────┘
echo.
%SSH% "[ -f ~/bloomi/DinoBloom-G.pth ] && rm -f ~/bloomi/DinoBloom-G.pth && echo '    Cleared DinoBloom-G.pth' || echo '    Not present — nothing to remove.'"
echo    NOTE: setup.sh downloads the original DinoBloom-G from Google Drive ^(gdown^).
echo.

:: ── [8] Run Remote Setup Script ───────────────────────────────────────────────
echo  ┌───────────────────────────────────────────────────────────────────────┐
echo  │  [8]   Run setup.sh on cluster  (installs OCI CLI, venv, packages)    │
echo  └───────────────────────────────────────────────────────────────────────┘
echo.
echo    Fixing line endings and running ~/setup.sh...
echo.
%SSH% "sed -i 's/\r//' ~/setup.sh && bash ~/setup.sh"
echo.

:: ── Done ──────────────────────────────────────────────────────────────────────
echo  ╔═══════════════════════════════════════════════════════════════════════╗
echo  ║          B L O O M  —  Setup Complete                                ║
echo  ╠═══════════════════════════════════════════════════════════════════════╣
echo  ║  Submit job  :  sbatch train_h200.sh                                 ║
echo  ║  Monitor     :  squeue -u rpatel1                                    ║
echo  ║  Watch logs  :  tail -f ~/bloomi/logs/dino_*.out                     ║
echo  ╚═══════════════════════════════════════════════════════════════════════╝
echo.

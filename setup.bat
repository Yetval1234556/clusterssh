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

set SCRIPTDIR=%~dp0
set SSH=ssh %CLUSTER_USER%@%CLUSTER_HOST%
set SCP=scp

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
pushd "%SCRIPTDIR%" && git pull origin master && popd
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

:: Check if agent is reachable
ssh-add -l >nul 2>&1
if errorlevel 1 (
    echo.
    echo  ╔═══════════════════════════════════════════════════════════════════════╗
    echo  ║  ACTION REQUIRED — SSH Agent is not running                          ║
    echo  ║                                                                       ║
    echo  ║  You will be asked for your passphrase on every SSH step.            ║
    echo  ║  To fix this permanently, run these TWO commands in PowerShell       ║
    echo  ║  as Administrator (right-click PowerShell ^> Run as administrator):  ║
    echo  ║                                                                       ║
    echo  ║    Set-Service ssh-agent -StartupType Automatic                      ║
    echo  ║    Start-Service ssh-agent                                            ║
    echo  ║                                                                       ║
    echo  ║  Then close this window and re-run setup.bat.                        ║
    echo  ╚═══════════════════════════════════════════════════════════════════════╝
    echo.
)
ssh-add %USERPROFILE%\.ssh\id_ed25519
if errorlevel 1 (
    echo    WARNING: Could not add key to agent. Passphrase will be required per step.
) else (
    echo    OK: SSH key loaded into agent — no more passphrase prompts this session.
)
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
%SCP% "%SCRIPTDIR%setup.sh" "%SCRIPTDIR%train_h200.sh" "%SCRIPTDIR%epoch_report.py" %CLUSTER_USER%@%CLUSTER_HOST%:~/
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
%SCP% "%SCRIPTDIR%train_efficientnet_b0.py" "%SCRIPTDIR%train_efficientnet_b0_ddp.py" "%SCRIPTDIR%requirements.txt" "%SCRIPTDIR%epoch_report.py" "%SCRIPTDIR%make_splits.py" %CLUSTER_USER%@%CLUSTER_HOST%:~/bloomi/
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
    %SCP% -r "%SCRIPTDIR%dinov2" %CLUSTER_USER%@%CLUSTER_HOST%:~/bloomi/
    if errorlevel 1 ( echo    ERROR: Failed to copy dinov2 — aborting. & exit /b 1 )
    echo    OK: dinov2 uploaded.
)
echo.

:: ── [4] Dataset: Google Drive folder → Oracle cluster (smart sync) ───────────
echo  ┌───────────────────────────────────────────────────────────────────────┐
echo  │  [4]   Dataset archives — smart check then sync                      │
echo  │        Check cluster first; only pull from Google Drive if missing   │
echo  └───────────────────────────────────────────────────────────────────────┘
echo.
echo    Cluster dest : ~/bloomi/New Data/extracted/
echo.

:: Check disk space first
echo    Disk space on cluster:
%SSH% "df -h ~ | tail -1 | awk '{printf \"    used=%%s  avail=%%s  (%%s full)\n\", $3, $4, $5}'"
echo.

:: Count images already on cluster
echo    Scanning cluster for existing archives and images...
for /f %%C in ('%SSH% "ls ~/bloomi/'New Data'/extracted/ 2>/dev/null | grep -cE '^archive[0-9]+' || echo 0"') do set ARCHIVE_COUNT=%%C
for /f %%I in ('%SSH% "find ~/bloomi/'New Data'/extracted/ -maxdepth 3 \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' -o -name '*.bmp' -o -name '*.tif' -o -name '*.tiff' \) 2>/dev/null | wc -l || echo 0"') do set IMAGE_COUNT=%%I
echo.
echo    Found on cluster: %ARCHIVE_COUNT% archives, %IMAGE_COUNT% images
echo.

:: Show current inventory
echo    ┌────────────────────────────────────────────────────────────────────┐
echo    │   Archive inventory                                                │
echo    ├────────────────────────────────────────────────────────────────────┤
%SSH% "ls ~/bloomi/'New Data'/extracted/ 2>/dev/null | grep -E '^archive[0-9]+' | sort -V | while read a; do IMG=$(find ~/bloomi/'New Data'/extracted/$a -maxdepth 2 \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' -o -name '*.bmp' \) 2>/dev/null | wc -l); SIZE=$(du -sh ~/bloomi/'New Data'/extracted/$a 2>/dev/null | cut -f1); echo \"    │   $a   ($IMG images, $SIZE)\"; done || echo '    │   (none found)'"
echo    └────────────────────────────────────────────────────────────────────┘
echo.

:: Decision: skip gdown if images already present
if %IMAGE_COUNT% GTR 0 (
    echo    SKIP: %IMAGE_COUNT% images already on cluster across %ARCHIVE_COUNT% archives.
    echo    Delete ~/bloomi/'New Data'/extracted/ on the cluster to force a re-sync.
    echo.
    goto :after_sync
)

:: No images found — attempt Google Drive download
echo    No images found on cluster. Attempting Google Drive sync...
echo    Google Drive : %GDRIVE_URL%
echo.

:: Ensure gdown is installed on cluster
echo    Ensuring gdown is installed on cluster...
%SSH% "python3 -c 'import gdown' 2>/dev/null && echo '    gdown already installed.' || ( (source ~/dinov2_venv/bin/activate 2>/dev/null && pip install -q gdown && echo '    gdown installed into venv.') || (pip install -q --break-system-packages gdown && echo '    gdown installed (system).') )"
echo.

:: Smart sync — gdown skips files already present
echo    Downloading from Google Drive (this may take a while)...
echo.
%SSH% "source ~/dinov2_venv/bin/activate 2>/dev/null; mkdir -p ~/bloomi/'New Data'/extracted && cd ~/bloomi/'New Data'/extracted && gdown --folder https://drive.google.com/drive/folders/%GDRIVE_ID% --remaining-ok && echo SYNC_COMPLETE"
if errorlevel 1 (
    echo.
    echo    WARNING: gdown sync failed.
    echo    The Google Drive folder may have per-file permission restrictions.
    echo.
    echo    To fix — ask the folder owner to set sharing to:
    echo      Google Drive ^> Right-click folder ^> Share ^> Anyone with the link ^> Viewer
    echo.
    echo    Or manually upload archives to the cluster:
    echo      scp -r "C:\path\to\archiveN" rpatel1@login-01.ncshare.org:~/bloomi/"New Data"/extracted/
    echo.
    echo    Continuing without dataset — training will fail if no images are present.
    echo.
    goto :after_sync
)
echo.

:after_sync
:: Final inventory
for /f %%C in ('%SSH% "ls ~/bloomi/'New Data'/extracted/ 2>/dev/null | grep -cE '^archive[0-9]+' || echo 0"') do set AFTER_COUNT=%%C
for /f %%I in ('%SSH% "find ~/bloomi/'New Data'/extracted/ -maxdepth 3 \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' -o -name '*.bmp' -o -name '*.tif' -o -name '*.tiff' \) 2>/dev/null | wc -l || echo 0"') do set AFTER_IMAGES=%%I
echo    Dataset status: %AFTER_COUNT% archives, %AFTER_IMAGES% images on cluster.
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
    %SCP% "%OCI_KEY%" %CLUSTER_USER%@%CLUSTER_HOST%:~/.oci/oci_api_key.pem
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

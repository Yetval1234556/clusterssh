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
::     - DinoBloom-G weights pulled from Oracle Object Storage (skips if already present)
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
pushd "%SCRIPTDIR%"
git remote get-url origin >nul 2>&1
if errorlevel 1 (
    echo    Setting remote origin to GitHub...
    git remote add origin https://github.com/Yetval1234556/clusterssh.git
)
git pull origin master
if errorlevel 1 (
    echo    WARNING: git pull failed - using local files.
) else (
    echo    OK: Repository is up to date.
)
popd
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
    echo  *** ACTION REQUIRED: ssh-agent is not running ***
    echo  You will be prompted for your passphrase on every SSH step.
    echo  To fix permanently, run these in PowerShell as Administrator:
    echo.
    echo    Set-Service ssh-agent -StartupType Automatic
    echo    Start-Service ssh-agent
    echo.
    echo  Then re-run setup.bat.
    echo.
)
ssh-add %USERPROFILE%\.ssh\id_ed25519
if errorlevel 1 (
    echo    WARNING: Could not add key to agent. Passphrase required per step.
) else (
    echo    OK: SSH key loaded into agent.
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

:: ── [4] Dataset: Pull from BOTH Oracle and Google Drive ──────────────────────
echo  ┌───────────────────────────────────────────────────────────────────────┐
echo  │  [4]   Dataset archives — pull from Oracle AND Google Drive           │
echo  │        Both sources are combined into extracted/                      │
echo  └───────────────────────────────────────────────────────────────────────┘
echo.

:: Disk space
echo    Disk space on cluster:
%SSH% "df -h ~ | tail -1 | awk '{printf \"    used=%%s  avail=%%s  (%%s full)\n\", $3, $4, $5}'"
echo.

:: ── [4a] Oracle ───────────────────────────────────────────────────────────────
echo    [4a] Pulling from Oracle (prefix: extracted/)...
echo    Downloading - each file will print as it lands:
echo.
%SSH% "export PATH=$HOME/.local/bin:$HOME/bin:$PATH; mkdir -p ~/bloomi/'New Data'; oci os object bulk-download --namespace idcsxwupyymi --bucket-name bloomi-training-data --prefix extracted/ --download-dir ~/bloomi/'New Data' --overwrite"
echo.

:: ── [4b] Google Drive ─────────────────────────────────────────────────────────
echo    [4b] Pulling from Google Drive (%GDRIVE_URL%)...
echo    (restricted files like labels.zip will be skipped automatically)
%SSH% "python3 -c 'import gdown' 2>/dev/null || (source ~/dinov2_venv/bin/activate 2>/dev/null && pip install -q gdown) || pip install -q --break-system-packages gdown"
%SSH% "source ~/dinov2_venv/bin/activate 2>/dev/null; mkdir -p ~/bloomi/'New Data'/extracted && cd ~/bloomi/'New Data'/extracted && gdown --folder https://drive.google.com/drive/folders/%GDRIVE_ID% --remaining-ok; exit 0"
echo.

:: Final count
for /f %%I in ('%SSH% "find ~/bloomi/'New Data'/extracted/ -maxdepth 5 \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' -o -name '*.bmp' -o -name '*.tif' -o -name '*.tiff' \) 2>/dev/null | wc -l || echo 0"') do set FINAL_IMAGES=%%I
if "%FINAL_IMAGES%"=="" set FINAL_IMAGES=0
echo    Total images on cluster after sync: %FINAL_IMAGES%
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

:: ── [7] Check DinoBloom-G weights ────────────────────────────────────────────
echo  ┌───────────────────────────────────────────────────────────────────────┐
echo  │  [7]   DinoBloom-G weights check                                      │
echo  │        Skip download if already present on cluster                   │
echo  └───────────────────────────────────────────────────────────────────────┘
echo.
%SSH% "if [ -f ~/bloomi/DinoBloom-G.pth ] && [ $(stat -c%%s ~/bloomi/DinoBloom-G.pth) -gt 104857600 ]; then echo '    OK: DinoBloom-G.pth already on cluster ('$(du -sh ~/bloomi/DinoBloom-G.pth | cut -f1)') - skipping download.'; else echo '    Not found or too small - setup.sh will download from Oracle.'; fi"
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

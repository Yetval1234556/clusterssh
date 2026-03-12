@echo off
setlocal

:: ── Config — update these ──────────────────────────────────────────────────────
set UNC_HOST=longleaf.unc.edu
set UNC_USER=rpatel1
:: ──────────────────────────────────────────────────────────────────────────────

:: Usage: setup.bat
:: Copies setup.sh to the cluster and runs it.

set SCRIPTDIR=%~dp0

echo Copying setup.sh to cluster...
scp "%SCRIPTDIR%setup.sh" %UNC_USER%@%UNC_HOST%:~/setup.sh
if errorlevel 1 (
    echo ERROR: SCP failed. Check your host and username in setup.bat.
    exit /b 1
)

echo Running setup on cluster...
ssh -o ConnectTimeout=10 %UNC_USER%@%UNC_HOST% "bash ~/setup.sh"

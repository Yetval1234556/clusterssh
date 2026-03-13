@echo off
setlocal

:: ── Config ────────────────────────────────────────────────────────────────────
set CLUSTER_HOST=login-01.ncshare.org
set CLUSTER_USER=rpatel1
:: ──────────────────────────────────────────────────────────────────────────────

:: Usage: .\setup.bat
:: Copies setup.sh to the cluster and runs it.

set SCRIPTDIR=%~dp0

echo Copying scripts to cluster...
scp "%SCRIPTDIR%setup.sh" "%SCRIPTDIR%train_h200.sh" "%SCRIPTDIR%epoch_report.py" %CLUSTER_USER%@%CLUSTER_HOST%:~/
if errorlevel 1 (
    echo ERROR: SCP failed. Check CLUSTER_HOST and CLUSTER_USER in setup.bat.
    exit /b 1
)

echo Running setup on cluster...
ssh -o ConnectTimeout=10 %CLUSTER_USER%@%CLUSTER_HOST% "sed -i 's/\r//' ~/setup.sh && bash ~/setup.sh"

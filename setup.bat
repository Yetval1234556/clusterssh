@echo off
setlocal

:: ── Config ────────────────────────────────────────────────────────────────────
set CLUSTER_HOST=login-01.ncshare.org
set CLUSTER_USER=rpatel1
set OCI_KEY=%~dp0yetvald@gmail.com-2026-03-13T02_48_12.614Z.pem
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

echo Copying OCI private key to cluster...
ssh -o ConnectTimeout=10 %CLUSTER_USER%@%CLUSTER_HOST% "mkdir -p ~/.oci && chmod 700 ~/.oci"
scp "%OCI_KEY%" %CLUSTER_USER%@%CLUSTER_HOST%:~/.oci/oci_api_key.pem
if errorlevel 1 (
    echo ERROR: Failed to copy OCI key. Check that the .pem file exists at:
    echo   %OCI_KEY%
    exit /b 1
)
ssh -o ConnectTimeout=10 %CLUSTER_USER%@%CLUSTER_HOST% "chmod 600 ~/.oci/oci_api_key.pem"
echo OCI key copied and secured.

echo Running setup on cluster...
ssh -o ConnectTimeout=10 %CLUSTER_USER%@%CLUSTER_HOST% "sed -i 's/\r//' ~/setup.sh && bash ~/setup.sh"

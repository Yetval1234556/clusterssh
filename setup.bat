@echo off
setlocal

:: ── Config ────────────────────────────────────────────────────────────────────
set ORACLE_HOST=YOUR_ORACLE_INSTANCE_IP
set ORACLE_USER=opc
:: ──────────────────────────────────────────────────────────────────────────────

:: Usage: .\setup.bat
:: Copies setup.sh to Oracle VM and runs it there.

set SCRIPTDIR=%~dp0

echo Copying setup.sh to Oracle VM...
scp "%SCRIPTDIR%setup.sh" %ORACLE_USER%@%ORACLE_HOST%:~/setup.sh
if errorlevel 1 (
    echo ERROR: SCP failed. Check ORACLE_HOST is set correctly in setup.bat.
    exit /b 1
)

echo Running setup on Oracle VM...
ssh -o ConnectTimeout=10 %ORACLE_USER%@%ORACLE_HOST% "sed -i 's/\r//' ~/setup.sh && bash ~/setup.sh"

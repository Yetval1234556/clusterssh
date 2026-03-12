@echo off
setlocal

:: ── Config — update these ──────────────────────────────────────────────────────
set ORACLE_HOST=YOUR_ORACLE_INSTANCE_IP
set ORACLE_USER=opc
:: ──────────────────────────────────────────────────────────────────────────────

:: Usage:
::   train_oracle_a100.bat      (1 GPU default)
::   train_oracle_a100.bat 2    (2 GPUs)

set NGPUS=%~1
if "%NGPUS%"=="" set NGPUS=1

set SCRIPTDIR=%~dp0

echo Copying train script to Oracle VM...
scp "%SCRIPTDIR%train_oracle_a100.sh" %ORACLE_USER%@%ORACLE_HOST%:~/train_oracle_a100.sh
if errorlevel 1 (
    echo ERROR: SCP failed. Check your host and username in train_oracle_a100.bat.
    exit /b 1
)

echo Starting training with %NGPUS% GPU(s) inside tmux...
ssh -o ConnectTimeout=10 %ORACLE_USER%@%ORACLE_HOST% "bash ~/train_oracle_a100.sh %NGPUS%"

echo.
echo Training is running in tmux session 'dinobloom' on the Oracle VM.
echo If your terminal closes, reconnect with:
echo   ssh %ORACLE_USER%@%ORACLE_HOST%
echo   tmux attach -t dinobloom
echo.
echo Monitor with:  monitor.bat oracle

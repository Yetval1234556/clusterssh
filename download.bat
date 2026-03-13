@echo off
setlocal

:: ── Config ────────────────────────────────────────────────────────────────────
set CLUSTER_HOST=login-01.ncshare.org
set CLUSTER_USER=rpatel1
set OCI_NS=idcsxwupyymi
set OCI_BUCKET=bloomi-training-data
:: ──────────────────────────────────────────────────────────────────────────────

:: Usage:
::   .\download.bat              — list all runs on Oracle
::   .\download.bat list         — list all runs on Oracle
::   .\download.bat best         — download best.pth from latest run
::   .\download.bat last         — download last.pth (checkpoint) from latest run
::   .\download.bat best JOB_ID  — download best.pth from a specific job folder

set MODE=%~1
set JOB=%~2
if "%MODE%"=="" set MODE=list

set SSH=ssh -o ConnectTimeout=10 %CLUSTER_USER%@%CLUSTER_HOST%
set OCI_CMD=oci os object list --namespace %OCI_NS% --bucket-name %OCI_BUCKET%

if "%MODE%"=="list" goto :list
if "%MODE%"=="best" goto :download
if "%MODE%"=="last" goto :download

echo Unknown mode: %MODE%
echo Usage: .\download.bat [list^|best^|last] [optional-job-folder]
exit /b 1

:: ── List all runs ─────────────────────────────────────────────────────────────
:list
echo.
echo All trained models in Oracle bucket:
echo.
%SSH% "oci os object list --namespace %OCI_NS% --bucket-name %OCI_BUCKET% --prefix trained-models/unc-h200/ --query 'data[].{Model:name,Size:\"size\"}' --output table 2>/dev/null"
echo.
echo To download:
echo   .\download.bat best              ^(latest run^)
echo   .\download.bat best job12345_20260313_120000  ^(specific run^)
goto :eof

:: ── Download ─────────────────────────────────────────────────────────────────
:download
echo.

:: Resolve job folder — use latest if not specified
if "%JOB%"=="" (
    echo Resolving latest run from Oracle...
    for /f "delims=" %%i in ('%SSH% "oci os object list --namespace %OCI_NS% --bucket-name %OCI_BUCKET% --prefix trained-models/unc-h200/ --query 'data[-1].name' --raw-output 2>/dev/null | sed 's|/[^/]*$||'"') do set FOLDER=%%i
) else (
    set FOLDER=trained-models/unc-h200/%JOB%
)

if "%FOLDER%"=="" (
    echo ERROR: Could not resolve a run folder from Oracle. Run .\download.bat list to see what's available.
    exit /b 1
)

set OBJNAME=%FOLDER%/%MODE%.pth
set OUTFILE=%~dp0%MODE%.pth

echo Downloading: %OBJNAME%
echo         To: %OUTFILE%
echo.

%SSH% "oci os object get --namespace %OCI_NS% --bucket-name %OCI_BUCKET% --name ""%OBJNAME%"" --file ~/downloads_%MODE%.pth && echo Download to cluster OK"
if errorlevel 1 (
    echo ERROR: Object not found on Oracle. Run .\download.bat list to see available runs.
    exit /b 1
)

scp %CLUSTER_USER%@%CLUSTER_HOST%:~/downloads_%MODE%.pth "%OUTFILE%"
if errorlevel 1 (
    echo ERROR: SCP from cluster to local failed.
    exit /b 1
)

echo.
echo Saved to: %OUTFILE%

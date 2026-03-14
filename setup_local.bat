@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: ── Config ────────────────────────────────────────────────────────────────────
set OCI_NS=idcsxwupyymi
set OCI_BUCKET=bloomi-training-data
set SCRIPTDIR=%~dp0
:: ──────────────────────────────────────────────────────────────────────────────

echo.
echo  ============================================================
echo    BLOOM -- Local Setup  (4070 Ti Super)
echo    Pulls data + DinoBloom-G from Oracle, sets up venv
echo  ============================================================
echo.

:: ── [0] Git Pull ──────────────────────────────────────────────────────────────
echo  [0] Pulling latest from GitHub...
pushd "%SCRIPTDIR%"
git remote get-url origin >nul 2>&1
if errorlevel 1 git remote add origin https://github.com/Yetval1234556/clusterssh.git
git pull origin master
if errorlevel 1 (
    echo    WARNING: git pull failed - using local files.
) else (
    echo    OK: up to date.
)
popd
echo.

:: ── [1] Python venv ───────────────────────────────────────────────────────────
echo  [1] Setting up Python venv...
if not exist "%SCRIPTDIR%.venv\Scripts\python.exe" (
    echo    Creating .venv...
    python -m venv "%SCRIPTDIR%.venv"
    if errorlevel 1 ( echo    ERROR: python -m venv failed. Is Python installed? & exit /b 1 )
    echo    OK: venv created.
) else (
    echo    SKIP: .venv already exists.
)
call "%SCRIPTDIR%.venv\Scripts\activate.bat"
echo    OK: venv activated.
echo.

:: ── [2] Install requirements ──────────────────────────────────────────────────
echo  [2] Installing requirements...
pip install -q -r "%SCRIPTDIR%requirements.txt"
if errorlevel 1 ( echo    ERROR: pip install failed. & exit /b 1 )
echo    OK: packages installed.
echo.

:: ── [3] Check OCI CLI ─────────────────────────────────────────────────────────
echo  [3] Checking OCI CLI...
where oci >nul 2>&1
if errorlevel 1 (
    echo    NOT FOUND - installing oci-cli...
    pip install -q oci-cli
    if errorlevel 1 ( echo    ERROR: oci-cli install failed. & exit /b 1 )
    echo    OK: oci-cli installed.
) else (
    echo    OK: oci CLI found.
)
echo.

:: ── [4] Dataset from Oracle ───────────────────────────────────────────────────
echo  [4] Checking dataset...
set DATA_DIR=%SCRIPTDIR%New Data
set EXTRACTED=%DATA_DIR%\extracted

if exist "%EXTRACTED%\ALL_NEW" (
    echo    SKIP: ALL_NEW already in New Data\extracted\ - not re-downloading.
    echo    Delete New Data\extracted\ALL_NEW to force re-download.
) else (
    echo    Downloading dataset from Oracle (this may take a while)...
    mkdir "%DATA_DIR%" >nul 2>&1
    oci os object bulk-download ^
        --namespace %OCI_NS% ^
        --bucket-name %OCI_BUCKET% ^
        --prefix extracted/ ^
        --download-dir "%DATA_DIR%" ^
        --overwrite
    if errorlevel 1 ( echo    ERROR: Dataset download failed. Check OCI config. & exit /b 1 )

    :: Remove nested extracted\extracted\ if it appeared
    if exist "%EXTRACTED%\extracted" (
        echo    Removing nested extracted\extracted\...
        rmdir /s /q "%EXTRACTED%\extracted"
    )
    echo    OK: Dataset downloaded.
)
echo.

:: ── [5] DinoBloom-G weights ───────────────────────────────────────────────────
echo  [5] Checking DinoBloom-G.pth...
set PTH=%SCRIPTDIR%DinoBloom-G.pth

if exist "%PTH%" (
    for %%F in ("%PTH%") do set SIZE=%%~zF
    if !SIZE! gtr 104857600 (
        echo    SKIP: DinoBloom-G.pth already present.
        goto :skip_pth
    )
)
echo    Downloading DinoBloom-G.pth from Oracle...
oci os object get ^
    --namespace %OCI_NS% ^
    --bucket-name %OCI_BUCKET% ^
    --name trained-models/dinobloom/DinoBloom-GDinoBloom-G.pth ^
    --file "%PTH%"
if errorlevel 1 ( echo    ERROR: DinoBloom-G.pth download failed. & exit /b 1 )
echo    OK: DinoBloom-G.pth downloaded.
:skip_pth
echo.

:: ── [6] Generate train/val splits ─────────────────────────────────────────────
echo  [6] Generating train.txt / val.txt splits...
python "%SCRIPTDIR%make_splits.py"
if errorlevel 1 (
    echo    WARNING: make_splits.py exited with error - check dataset.
)
echo.

:: ── Done ──────────────────────────────────────────────────────────────────────
echo  ============================================================
echo    Setup complete!
echo.
echo    To start training:
echo      train_local.bat
echo.
echo    Or manually:
echo      python train_efficientnet_b0.py --epochs 30 --no-oracle
echo  ============================================================
echo.
pause

@echo off
chcp 65001 >nul
setlocal

echo.
echo  ============================================================
echo    BLOOM -- Local Training (4070 Ti Super)
echo    Epochs  : 30
echo    Oracle  : OFF (saves locally only)
echo    Output  : bloom_leukemia.pth + checkpoint_latest.pth
echo  ============================================================
echo.

:: ── Activate venv ────────────────────────────────────────────────────────────
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo  Venv activated.
) else if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo  Venv activated.
) else (
    echo  No venv found - using system Python.
)
echo.

:: ── Run training ─────────────────────────────────────────────────────────────
cd /d "%~dp0"
python train_efficientnet_b0.py ^
    --epochs          30 ^
    --lr              1e-4 ^
    --unfreeze-blocks 2 ^
    --workers         4 ^
    --report-every    5 ^
    --no-oracle

echo.
echo  ============================================================
echo    Done. Models saved locally:
echo      bloom_leukemia.pth     = best val acc
echo      checkpoint_latest.pth  = last epoch (resume)
echo  ============================================================
echo.
pause

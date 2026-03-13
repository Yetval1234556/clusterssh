#!/bin/bash
#SBATCH --job-name=dinobloom-h200
#SBATCH --output=/hpc/home/%u/bloomi/logs/dino_%j.out
#SBATCH --error=/hpc/home/%u/bloomi/logs/dino_%j.err
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gres=gpu:4
#SBATCH --constraint=h200
#SBATCH --mem=900G
#SBATCH --time=96:00:00
#SBATCH --nice=0

# Usage:
#   sbatch train_h200.sh       (4 GPUs default)
#   sbatch train_h200.sh 2     (2 GPUs)
#   sbatch train_h200.sh 1     (1 GPU)

set -e

NGPUS=${1:-4}
RUN_DATE=$(date +%Y%m%d_%H%M%S)
OCI_PREFIX="trained-models/unc-h200/job${SLURM_JOB_ID}_${RUN_DATE}"
OCI_NS="idcsxwupyymi"
OCI_BUCKET="bloomi-training-data"

# ── Environment ───────────────────────────────────────────────────────────────
source $HOME/.bashrc
source "$HOME/dinov2_venv/bin/activate"

SCRATCH=/hpc/home/$USER/bloomi
cd $SCRATCH
mkdir -p logs output

# Copy epoch reporter into working dir so training scripts can import it
cp "$HOME/epoch_report.py" "$SCRATCH/epoch_report.py" 2>/dev/null \
    && echo "epoch_report.py ready in $SCRATCH" \
    || echo "WARNING: epoch_report.py not found in ~/ — run setup.bat first"

# ── Job Info ──────────────────────────────────────────────────────────────────
echo "========================================================"
echo "  DinoBloom-G Fine-Tuning — UNC H200 Cluster"
echo "========================================================"
echo "  Job ID    : $SLURM_JOB_ID"
echo "  Node      : $(hostname)"
echo "  Date      : $(date)"
echo "  GPUs      : ${NGPUS}x H200 (96GB VRAM each)"
echo "  Batch/GPU : 64  |  Effective batch: $((64 * NGPUS))"
echo "  Epochs    : 75  |  LR: 1e-4  |  Unfreeze: 4 blocks"
echo "  Workers   : 224  |  Report every: 5 epochs"
echo "  Scratch   : $SCRATCH"
echo "  Oracle    : oci://$OCI_BUCKET/$OCI_PREFIX"
echo "========================================================"
echo ""

echo "--- Python / PyTorch ---"
echo "  Python  : $(python --version 2>&1)"
echo "  PyTorch : $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not found')"
echo "  CUDA    : $(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'not found')"
echo "  Venv    : $VIRTUAL_ENV"
echo "  nvcc    : $(nvcc --version 2>/dev/null | grep release || echo 'not in PATH')"
echo ""

echo "--- GPU Hardware ---"
nvidia-smi
echo ""

echo "--- Disk Space ---"
df -h $SCRATCH
echo ""

# ── Distributed setup ─────────────────────────────────────────────────────────
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * NGPUS))

echo "--- Distributed Config ---"
echo "  Master   : $MASTER_ADDR:$MASTER_PORT"
echo "  World    : $WORLD_SIZE ranks"
echo "  Nodes    : $SLURM_NNODES  |  GPUs/node: $NGPUS"
echo ""

# ── Background GPU monitor (every 30s) ────────────────────────────────────────
nvidia-smi \
    --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.free,temperature.gpu,power.draw \
    --format=csv --loop=30 > logs/gpu_monitor_${SLURM_JOB_ID}.csv &
GPU_MONITOR_PID=$!
echo "GPU monitor PID $GPU_MONITOR_PID → logs/gpu_monitor_${SLURM_JOB_ID}.csv"
echo ""

# ── Background checkpoint backup to Oracle (every 30 min) ────────────────────
# Runs silently in background — if job crashes, latest backup is safe on Oracle
(
    BACKUP_PREFIX="trained-models/unc-h200/job${SLURM_JOB_ID}_${RUN_DATE}/backups"
    INTERVAL=1800  # 30 minutes
    N=0
    while true; do
        sleep $INTERVAL
        N=$((N + 1))
        TS=$(date +%H%M%S)
        BACKED_UP=0

        if [ -f "$SCRATCH/checkpoint_latest.pth" ]; then
            oci os object put \
                --namespace $OCI_NS --bucket-name $OCI_BUCKET \
                --name "$BACKUP_PREFIX/checkpoint_${TS}.pth" \
                --file "$SCRATCH/checkpoint_latest.pth" --force \
                > /dev/null 2>&1 && BACKED_UP=1
        fi

        if [ -f "$SCRATCH/dinobloom_g_finetuned.pth" ]; then
            oci os object put \
                --namespace $OCI_NS --bucket-name $OCI_BUCKET \
                --name "$BACKUP_PREFIX/best_${TS}.pth" \
                --file "$SCRATCH/dinobloom_g_finetuned.pth" --force \
                > /dev/null 2>&1
        fi

        if [ $BACKED_UP -eq 1 ]; then
            echo "[backup #$N @ $(date)] Checkpoint pushed → $BACKUP_PREFIX/checkpoint_${TS}.pth"
        else
            echo "[backup #$N @ $(date)] No checkpoint found yet — skipping"
        fi
    done
) &
BACKUP_PID=$!
echo "Checkpoint backup PID $BACKUP_PID → Oracle every 30min under backups/"
echo ""

# ── Train ─────────────────────────────────────────────────────────────────────
echo "========================================================"
echo "  Starting training — $(date)"
echo "========================================================"

if [ "$NGPUS" -eq 1 ]; then
    python train_efficientnet_b0.py \
        --epochs 75 \
        --batch-size 64 \
        --lr 1e-4 \
        --unfreeze-blocks 4 \
        --workers 224 \
        --report-every 5
else
    echo "  DDP: torchrun across $NGPUS GPUs (effective batch $((64 * NGPUS)))"
    srun torchrun \
        --nnodes=$SLURM_NNODES \
        --nproc_per_node=$NGPUS \
        --rdzv_id=$SLURM_JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        train_efficientnet_b0_ddp.py \
            --epochs 75 \
            --batch-size 64 \
            --lr 1e-4 \
            --unfreeze-blocks 4 \
            --workers 224 \
            --report-every 5
fi

# ── Stop background processes ─────────────────────────────────────────────────
kill $GPU_MONITOR_PID 2>/dev/null || true
kill $BACKUP_PID 2>/dev/null || true

echo ""
echo "========================================================"
echo "  Training done — $(date)"
echo "========================================================"
nvidia-smi
echo ""

# ── Upload to Oracle ──────────────────────────────────────────────────────────
echo "--- Uploading models to Oracle Object Storage ---"
echo "  Bucket : $OCI_BUCKET"
echo "  Folder : $OCI_PREFIX"
echo ""

# Write a run_info.txt so each Oracle folder is self-describing
BEST_SIZE=$(du -sh "$SCRATCH/dinobloom_g_finetuned.pth" 2>/dev/null | cut -f1 || echo "not found")
LAST_SIZE=$(du -sh "$SCRATCH/checkpoint_latest.pth" 2>/dev/null | cut -f1 || echo "not found")

# Build per-5-epoch metrics table from training_metrics.csv
METRICS_TABLE=""
if [ -f "$SCRATCH/training_metrics.csv" ]; then
    HEADER=$(head -1 "$SCRATCH/training_metrics.csv")
    METRICS_TABLE="$HEADER"$'\n'
    # Every 5th epoch (rows where epoch % 5 == 0), plus epoch 75 always
    METRICS_TABLE+=$(awk -F',' 'NR>1 && ($1 % 5 == 0 || $1 == 75) {print}' "$SCRATCH/training_metrics.csv")
else
    METRICS_TABLE="no training_metrics.csv found"
fi

cat > /tmp/run_info.txt << INFO
================================================================================
  BLOOMI — DinoBloom-G Fine-Tuning Run
  Leukemia Subtype Classifier | UNC H200 Cluster
================================================================================

PROJECT
-------
Name        : Bloomi
Goal        : Fine-tune DinoBloom-G (Vision Transformer, DINOv2-Giant backbone)
              to classify leukemia subtypes from white blood cell microscopy images.
Application : Early leukemia detection — model predicts malignant cell subtypes
              (Early, Pre, Pro, Blast) from bone marrow / peripheral blood smear
              images. Intended for clinical decision support.
Base Model  : DinoBloom-G — a ViT-Giant pretrained on 13M+ pathology images,
              specifically on hematology slides. Fine-tuned here on our curated
              WBC malignancy dataset using EfficientNet-B0 classification head.
Dataset     : WBC Malignancy Dataset (~10GB)
              Classes: Early, Pre, Pro, Blast (malignant subtypes)
              Source : Oracle bucket bloomi-training-data/extracted/

TRAINING RUN
------------
Job ID      : $SLURM_JOB_ID
Date        : $RUN_DATE
Node        : $(hostname)
Cluster     : UNC ncshare H200 HPC
GPUs        : ${NGPUS}x NVIDIA H200 (96GB VRAM each = $((96 * NGPUS))GB total)
Epochs      : 75
Batch/GPU   : 64  (effective batch: $((64 * NGPUS)))
Learning Rate: 1e-4
Unfreeze    : Last 4 blocks of DinoBloom-G backbone (rest frozen)
Workers     : 224 dataloader workers
Strategy    : DDP (DistributedDataParallel) via torchrun across $NGPUS GPUs

WHY THESE SETTINGS
------------------
- DinoBloom-G frozen except last 4 blocks: preserves hematology-specific
  features learned during pretraining, only adapts top layers to our classes.
- LR 1e-4: conservative rate suited for fine-tuning a large pretrained ViT.
- 75 epochs: enough to converge without overfitting on our dataset size.
- Batch 256 effective: large batch stabilises DDP gradient averaging across GPUs.

FILES IN THIS FOLDER
--------------------
best.pth       : $BEST_SIZE  — checkpoint at best validation accuracy (use for inference)
last.pth       : $LAST_SIZE  — final epoch checkpoint (use to resume training)
run_info.txt   : this file
backups/       : mid-training checkpoints saved every 30 min (crash recovery)

METRICS — every 5 epochs (5, 10, 15 ... 75)
--------------------------------------------
$METRICS_TABLE

ORACLE STORAGE PATH
-------------------
Namespace : $OCI_NS
Bucket    : $OCI_BUCKET
Prefix    : $OCI_PREFIX/
================================================================================
INFO
oci os object put \
    --namespace $OCI_NS --bucket-name $OCI_BUCKET \
    --name "$OCI_PREFIX/run_info.txt" \
    --file /tmp/run_info.txt --force
echo "  Uploaded → $OCI_PREFIX/run_info.txt"
echo ""

if [ -f "$SCRATCH/dinobloom_g_finetuned.pth" ]; then
    SIZE=$(du -sh "$SCRATCH/dinobloom_g_finetuned.pth" | cut -f1)
    echo "  best.pth ($SIZE) → uploading..."
    oci os object put \
        --namespace $OCI_NS --bucket-name $OCI_BUCKET \
        --name "$OCI_PREFIX/best.pth" \
        --file "$SCRATCH/dinobloom_g_finetuned.pth" --force
    echo "  Uploaded → $OCI_PREFIX/best.pth"
    rm -f "$SCRATCH/dinobloom_g_finetuned.pth"
    echo "  Local copy removed."
else
    echo "  WARNING: dinobloom_g_finetuned.pth not found — skipping best upload"
fi

if [ -f "$SCRATCH/checkpoint_latest.pth" ]; then
    SIZE=$(du -sh "$SCRATCH/checkpoint_latest.pth" | cut -f1)
    echo "  last.pth ($SIZE) → uploading..."
    oci os object put \
        --namespace $OCI_NS --bucket-name $OCI_BUCKET \
        --name "$OCI_PREFIX/last.pth" \
        --file "$SCRATCH/checkpoint_latest.pth" --force
    echo "  Uploaded → $OCI_PREFIX/last.pth"
    rm -f "$SCRATCH/checkpoint_latest.pth"
    echo "  Local copy removed."
else
    echo "  WARNING: checkpoint_latest.pth not found — skipping last upload"
fi

echo ""
echo "--- Verifying uploads on Oracle ---"
oci os object list \
    --namespace $OCI_NS \
    --bucket-name $OCI_BUCKET \
    --prefix "$OCI_PREFIX/" \
    --query 'data[].{name:name, size:"size"}' \
    --output table
echo ""
echo "--- All runs stored in Oracle ---"
oci os object list \
    --namespace $OCI_NS \
    --bucket-name $OCI_BUCKET \
    --prefix "trained-models/unc-h200/" \
    --query 'data[].{name:name, size:"size"}' \
    --output table

echo ""
echo "========================================================"
echo "  All done — $(date)"
echo "  Models at: oci://$OCI_BUCKET/$OCI_PREFIX"
echo ""
echo "  Download best model:"
echo "    oci os object get --namespace $OCI_NS --bucket-name $OCI_BUCKET \\"
echo "      --name \"$OCI_PREFIX/best.pth\" --file ~/dinobloom_finetuned.pth"
echo "========================================================"

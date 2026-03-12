#!/bin/bash
# ── Oracle A100 Training Script ────────────────────────────────────────────────
# Run directly on the Oracle A100 instance (no SLURM).
# Usage: bash train_oracle_a100.sh [num_gpus]
#   bash train_oracle_a100.sh 1   (single A100 80GB)
#   bash train_oracle_a100.sh 2   (if using VM.GPU.A100.2)

set -e

NGPUS=${1:-1}
SCRATCH=$HOME/bloomi

# ── Environment ───────────────────────────────────────────────────────────────
source $HOME/.bashrc
conda activate dinov2

cd $SCRATCH
mkdir -p logs

# ── Info ──────────────────────────────────────────────────────────────────────
echo "========================================================"
echo "  DinoBloom-G Fine-Tuning — Oracle A100"
echo "========================================================"
echo "  GPUs      : ${NGPUS}x A100 (80GB VRAM each)"
echo "  Host      : $(hostname)"
echo "  Date      : $(date)"
echo "  CUDA      : $(nvcc --version 2>/dev/null | grep release || echo 'nvcc not in PATH')"
echo "  Batch/GPU : 64  |  Effective batch: $((64 * NGPUS))"
echo "  Epochs    : 75"
echo "========================================================"
echo ""
nvidia-smi
echo ""

# ── Background GPU monitor (every 30s) ────────────────────────────────────────
LOGFILE=logs/gpu_monitor_$(date +%Y%m%d_%H%M%S).csv
nvidia-smi \
    --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.free,temperature.gpu,power.draw \
    --format=csv --loop=30 > $LOGFILE &
GPU_MONITOR_PID=$!
echo "GPU monitor PID $GPU_MONITOR_PID → $LOGFILE"
echo ""

# ── Train ─────────────────────────────────────────────────────────────────────
echo "=== Starting training ==="

if [ "$NGPUS" -eq 1 ]; then
    python train_efficientnet_b0.py \
        --epochs 75 \
        --batch-size 64 \
        --lr 1e-4 \
        --unfreeze-blocks 4 \
        --workers 8
else
    echo "Effective batch size: $((64 * NGPUS)) (64 per GPU x $NGPUS GPUs)"
    torchrun --nproc_per_node=$NGPUS \
        train_efficientnet_b0_ddp.py \
            --epochs 75 \
            --batch-size 64 \
            --lr 1e-4 \
            --unfreeze-blocks 4 \
            --workers 8
fi

# ── Stop GPU monitor ──────────────────────────────────────────────────────────
kill $GPU_MONITOR_PID 2>/dev/null || true

# ── Final GPU state ───────────────────────────────────────────────────────────
echo ""
echo "=== Final GPU state ==="
nvidia-smi
echo "=== Training done: $(date) ==="

# ── Upload model to Oracle Object Storage ─────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DEST="trained-models/dinobloom_g_a100_${TIMESTAMP}.pth"

echo ""
echo "=== Uploading model to Oracle Object Storage ==="
oci os object put \
    --namespace idcsxwupyymi \
    --bucket-name bloomi-training-data \
    --name "$DEST" \
    --file $SCRATCH/dinobloom_g_finetuned.pth \
    --force

echo "=== Model uploaded to: bloomi-training-data/$DEST ==="

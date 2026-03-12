#!/bin/bash
#SBATCH -o /scratch/%u/bloomi/logs/run_%j.txt
#SBATCH -e /scratch/%u/bloomi/logs/error_%j.txt
#SBATCH -J dino-train
#SBATCH --reservation=test_supergpu05
#SBATCH --qos gpu_reservation
#SBATCH -p gpu_p
#SBATCH --time=96:00:00
#SBATCH --nice=10000

# ── Usage ─────────────────────────────────────────────────────────────────────
# Single GPU:  sbatch --gres=gpu:1 -c 16  --mem=128G  train_cluster.sh 1
# 8 GPUs:      sbatch --gres=gpu:8 -c 126 --mem=1800G train_cluster.sh 8

set -e

NGPUS=${1:-1}   # Number of GPUs — pass as argument, defaults to 1

# ── Environment ───────────────────────────────────────────────────────────────
source $HOME/.bashrc
conda activate dinov2

SCRATCH=/scratch/$USER/bloomi
cd $SCRATCH
mkdir -p logs

# ── GPU Info ──────────────────────────────────────────────────────────────────
echo "========================================================"
echo "  DinoBloom-G Fine-Tuning"
echo "========================================================"
echo "  Mode      : ${NGPUS} GPU(s)"
echo "  Node      : $(hostname)"
echo "  Date      : $(date)"
echo "  Job ID    : $SLURM_JOB_ID"
echo "  CUDA      : $(nvcc --version 2>/dev/null | grep release || echo 'nvcc not in PATH')"
echo "========================================================"
echo ""
nvidia-smi
echo ""

# ── Background GPU monitor (logs every 30s) ───────────────────────────────────
nvidia-smi \
    --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.free,temperature.gpu,power.draw \
    --format=csv --loop=30 > logs/gpu_monitor_${SLURM_JOB_ID}.csv &
GPU_MONITOR_PID=$!
echo "GPU monitor PID $GPU_MONITOR_PID → logs/gpu_monitor_${SLURM_JOB_ID}.csv"
echo ""

# ── Train ─────────────────────────────────────────────────────────────────────
echo "=== Starting training ==="

if [ "$NGPUS" -eq 1 ]; then
    # Single GPU
    python train_efficientnet_b0.py \
        --epochs 30 \
        --batch-size 8 \
        --lr 1e-4 \
        --unfreeze-blocks 4 \
        --workers 16
else
    # Multi-GPU — effective batch = batch_size x NGPUS
    echo "Effective batch size: $((8 * NGPUS)) (8 per GPU x $NGPUS GPUs)"
    torchrun --nproc_per_node=$NGPUS train_efficientnet_b0_ddp.py \
        --epochs 30 \
        --batch-size 8 \
        --lr 1e-4 \
        --unfreeze-blocks 4 \
        --workers 16
fi

# ── Stop GPU monitor ──────────────────────────────────────────────────────────
kill $GPU_MONITOR_PID 2>/dev/null || true

# ── Final GPU state ───────────────────────────────────────────────────────────
echo ""
echo "=== Final GPU state ==="
nvidia-smi
echo ""
echo "=== Training done: $(date) ==="

# ── Upload model to Oracle Object Storage ─────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DEST="trained-models/dinobloom_g_leukemia_classifier_${TIMESTAMP}.pth"

echo ""
echo "=== Uploading model to Oracle Object Storage ==="
oci os object put \
    --namespace idcsxwupyymi \
    --bucket-name bloomi-training-data \
    --name "$DEST" \
    --file $SCRATCH/dinobloom_g_finetuned.pth \
    --force

echo "=== Model uploaded to: bloomi-training-data/$DEST ==="

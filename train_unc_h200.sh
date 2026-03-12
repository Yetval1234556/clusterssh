#!/bin/bash
#SBATCH --job-name=dinobloom-h200
#SBATCH --output=/scratch/%u/bloomi/logs/dino_%j.out
#SBATCH --error=/scratch/%u/bloomi/logs/dino_%j.err
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --gres=gpu:8
#SBATCH --constraint=h200
#SBATCH --mem=1800G
#SBATCH --time=96:00:00
#SBATCH --nice=10000

# ── Usage ─────────────────────────────────────────────────────────────────────
# sbatch train_unc_h200.sh       (8 GPUs default)
# sbatch train_unc_h200.sh 1     (1 GPU)

set -e

NGPUS=${1:-8}

# ── Environment ───────────────────────────────────────────────────────────────
source $HOME/.bashrc
# source $HOME/miniconda3/etc/profile.d/conda.sh  # uncomment if needed
conda activate dinov2

SCRATCH=/scratch/$USER/bloomi
cd $SCRATCH
mkdir -p logs output

# ── Info ──────────────────────────────────────────────────────────────────────
echo "========================================================"
echo "  DinoBloom-G Fine-Tuning — UNC H200 Cluster"
echo "========================================================"
echo "  GPUs      : ${NGPUS}x H200 (96GB VRAM each)"
echo "  Node      : $(hostname)"
echo "  Job ID    : $SLURM_JOB_ID"
echo "  Date      : $(date)"
echo "  CUDA      : $(nvcc --version 2>/dev/null | grep release || echo 'nvcc not in PATH')"
echo "  Batch/GPU : 64  |  Effective batch: $((64 * NGPUS))"
echo "  Epochs    : 75"
echo "========================================================"
echo ""
nvidia-smi
echo ""

# ── Distributed setup ─────────────────────────────────────────────────────────
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
NUM_GPUS=$(nvidia-smi -L | wc -l)
export WORLD_SIZE=$((SLURM_NNODES * NUM_GPUS))

echo "Master Addr : $MASTER_ADDR"
echo "Master Port : $MASTER_PORT"
echo "World Size  : $WORLD_SIZE"
echo "GPUs/Node   : $NUM_GPUS"
echo ""

# ── Background GPU monitor (every 30s) ────────────────────────────────────────
nvidia-smi \
    --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.free,temperature.gpu,power.draw \
    --format=csv --loop=30 > logs/gpu_monitor_${SLURM_JOB_ID}.csv &
GPU_MONITOR_PID=$!
echo "GPU monitor PID $GPU_MONITOR_PID → logs/gpu_monitor_${SLURM_JOB_ID}.csv"
echo ""

# ── Train ─────────────────────────────────────────────────────────────────────
echo "=== Starting training ==="

if [ "$NGPUS" -eq 1 ]; then
    python train_efficientnet_b0.py \
        --epochs 75 \
        --batch-size 64 \
        --lr 1e-4 \
        --unfreeze-blocks 4 \
        --workers 112
else
    echo "Effective batch size: $((64 * NGPUS)) (64 per GPU x $NGPUS GPUs)"
    srun torchrun \
        --nnodes=$SLURM_NNODES \
        --nproc_per_node=$NUM_GPUS \
        --rdzv_id=$SLURM_JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        train_efficientnet_b0_ddp.py \
            --epochs 75 \
            --batch-size 64 \
            --lr 1e-4 \
            --unfreeze-blocks 4 \
            --workers 112
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
DEST="trained-models/dinobloom_g_h200_${TIMESTAMP}.pth"

echo ""
echo "=== Uploading model to Oracle Object Storage ==="
oci os object put \
    --namespace idcsxwupyymi \
    --bucket-name bloomi-training-data \
    --name "$DEST" \
    --file $SCRATCH/dinobloom_g_finetuned.pth \
    --force

echo "=== Model uploaded to: bloomi-training-data/$DEST ==="

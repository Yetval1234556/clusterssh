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
RUN_DATE=$(date +%Y%m%d_%H%M%S)
OCI_PREFIX="trained-models/unc-h200/job${SLURM_JOB_ID}_${RUN_DATE}"
OCI_NS="idcsxwupyymi"
OCI_BUCKET="bloomi-training-data"

# ── Environment ───────────────────────────────────────────────────────────────
source $HOME/.bashrc
# Source conda init from common locations (needed in non-interactive SLURM shells)
for f in "$HOME/miniconda3/etc/profile.d/conda.sh" \
          "$HOME/anaconda3/etc/profile.d/conda.sh" \
          "/opt/conda/etc/profile.d/conda.sh"; do
    [ -f "$f" ] && source "$f" && break
done
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
export WORLD_SIZE=$((SLURM_NNODES * NGPUS))

echo "Master Addr : $MASTER_ADDR"
echo "Master Port : $MASTER_PORT"
echo "World Size  : $WORLD_SIZE"
echo "GPUs/Node   : $NGPUS"
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
        --nproc_per_node=$NGPUS \
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

# ── Upload best and last to Oracle ────────────────────────────────────────────
echo "=== Uploading models to Oracle Object Storage ==="
echo "  Run folder: $OCI_PREFIX"

if [ -f "$SCRATCH/dinobloom_g_finetuned.pth" ]; then
    oci os object put \
        --namespace $OCI_NS --bucket-name $OCI_BUCKET \
        --name "$OCI_PREFIX/best.pth" \
        --file "$SCRATCH/dinobloom_g_finetuned.pth" --force
    echo "  Uploaded → $OCI_PREFIX/best.pth"
else
    echo "  WARNING: dinobloom_g_finetuned.pth not found — skipping best upload"
fi

if [ -f "$SCRATCH/checkpoint_latest.pth" ]; then
    oci os object put \
        --namespace $OCI_NS --bucket-name $OCI_BUCKET \
        --name "$OCI_PREFIX/last.pth" \
        --file "$SCRATCH/checkpoint_latest.pth" --force
    echo "  Uploaded → $OCI_PREFIX/last.pth"
else
    echo "  WARNING: checkpoint_latest.pth not found — skipping last upload"
fi

echo "=== Upload done ==="

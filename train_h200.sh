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
#SBATCH --nice=10000

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
for f in "$HOME/miniconda3/etc/profile.d/conda.sh" \
          "$HOME/anaconda3/etc/profile.d/conda.sh" \
          "/opt/conda/etc/profile.d/conda.sh"; do
    [ -f "$f" ] && source "$f" && break
done
conda activate dinov2

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
echo "  Conda   : $CONDA_DEFAULT_ENV"
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

# ── Stop GPU monitor ──────────────────────────────────────────────────────────
kill $GPU_MONITOR_PID 2>/dev/null || true

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

if [ -f "$SCRATCH/dinobloom_g_finetuned.pth" ]; then
    SIZE=$(du -sh "$SCRATCH/dinobloom_g_finetuned.pth" | cut -f1)
    echo "  best.pth ($SIZE) → uploading..."
    oci os object put \
        --namespace $OCI_NS --bucket-name $OCI_BUCKET \
        --name "$OCI_PREFIX/best.pth" \
        --file "$SCRATCH/dinobloom_g_finetuned.pth" --force
    echo "  Uploaded → $OCI_PREFIX/best.pth"
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
else
    echo "  WARNING: checkpoint_latest.pth not found — skipping last upload"
fi

echo ""
echo "========================================================"
echo "  All done — $(date)"
echo "  Models at: oci://$OCI_BUCKET/$OCI_PREFIX"
echo "========================================================"

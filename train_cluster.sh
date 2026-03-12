#!/bin/bash
#SBATCH --job-name=dinobloom-finetune
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=gpu          # UPDATE: set to your cluster's GPU partition name

# ── Environment ───────────────────────────────────────────────────────────────
module load cuda/11.7 2>/dev/null || true
module load conda 2>/dev/null || true
conda activate dinov2

SCRATCH=/scratch/$USER/bloomi

cd $SCRATCH
mkdir -p logs

echo "=== Starting DinoBloom-G fine-tuning ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Time: $(date)"

# ── Train ─────────────────────────────────────────────────────────────────────
python train_efficientnet_b0.py \
    --epochs 30 \
    --batch-size 16 \
    --lr 1e-4 \
    --unfreeze-blocks 4 \
    --workers 8

echo "=== Training complete: $(date) ==="

#!/bin/bash
#SBATCH -o logs/run_%j.txt
#SBATCH -e logs/error_%j.txt
#SBATCH -J dino-1gpu
#SBATCH --reservation=test_supergpu05
#SBATCH --qos gpu_reservation
#SBATCH -p gpu_p
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH --time=96:00:00
#SBATCH --nice=10000
#SBATCH --gres=gpu:1

# ── Environment ───────────────────────────────────────────────────────────────
source $HOME/.bashrc
conda activate dinov2
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

SCRATCH=/scratch/$USER/bloomi
cd $SCRATCH
mkdir -p logs

echo "=== DinoBloom-G Fine-Tuning (1 GPU) ==="
echo "Node : $(hostname)"
echo "GPU  : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Time : $(date)"

python train_efficientnet_b0.py \
    --epochs 30 \
    --batch-size 8 \
    --lr 1e-4 \
    --unfreeze-blocks 4 \
    --workers 16

echo "=== Done: $(date) ==="

#!/bin/bash
#SBATCH -o /scratch/%u/bloomi/logs/run_%j.txt
#SBATCH -e /scratch/%u/bloomi/logs/error_%j.txt
#SBATCH -J dino-8gpu
#SBATCH --reservation=test_supergpu05
#SBATCH --qos gpu_reservation
#SBATCH -p gpu_p
#SBATCH -c 126
#SBATCH --mem=1800G
#SBATCH --time=96:00:00
#SBATCH --nice=10000
#SBATCH --gres=gpu:8

set -e  # Exit immediately if any command fails

# ── Environment ───────────────────────────────────────────────────────────────
source $HOME/.bashrc
conda activate dinov2

SCRATCH=/scratch/$USER/bloomi
cd $SCRATCH
mkdir -p logs

# ── GPU Info ──────────────────────────────────────────────────────────────────
echo "=== DinoBloom-G Fine-Tuning (8 GPUs) ==="
echo "Node      : $(hostname)"
echo "Date      : $(date)"
echo "CUDA      : $(nvcc --version 2>/dev/null | grep release || echo 'nvcc not in PATH')"
echo ""
nvidia-smi
echo ""

# ── Train ─────────────────────────────────────────────────────────────────────
echo "=== Starting training ==="
# Effective batch size = 8 (per GPU) x 8 (GPUs) = 64
torchrun --nproc_per_node=8 train_efficientnet_b0_ddp.py \
    --epochs 30 \
    --batch-size 8 \
    --lr 1e-4 \
    --unfreeze-blocks 4 \
    --workers 16

# ── GPU stats after training ──────────────────────────────────────────────────
echo ""
echo "=== Final GPU state ==="
nvidia-smi
echo "=== Training done: $(date) ==="

# ── Upload model to Oracle Object Storage ─────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DEST="trained-models/dinobloom_g_leukemia_classifier_${TIMESTAMP}.pth"

echo "=== Uploading model to Oracle Object Storage ==="
oci os object put \
    --namespace idcsxwupyymi \
    --bucket-name bloomi-training-data \
    --name "$DEST" \
    --file $SCRATCH/dinobloom_g_finetuned.pth \
    --force

echo "=== Model uploaded to: bloomi-training-data/$DEST ==="

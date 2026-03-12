#!/bin/bash
#SBATCH -o logs/run_%j.txt
#SBATCH -e logs/error_%j.txt
#SBATCH -J dino-8gpu
#SBATCH --reservation=test_supergpu05
#SBATCH --qos gpu_reservation
#SBATCH -p gpu_p
#SBATCH -c 126
#SBATCH --mem=1800G
#SBATCH --time=96:00:00
#SBATCH --nice=10000
#SBATCH --gres=gpu:8

# ── Environment ───────────────────────────────────────────────────────────────
source $HOME/.bashrc
conda activate dinov2
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

SCRATCH=/scratch/$USER/bloomi
cd $SCRATCH
mkdir -p logs

echo "=== DinoBloom-G Fine-Tuning (8 GPUs) ==="
echo "Node : $(hostname)"
echo "GPUs : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Time : $(date)"

# Effective batch size = 8 (per GPU) x 8 (GPUs) = 64
torchrun --nproc_per_node=8 train_efficientnet_b0_ddp.py \
    --epochs 30 \
    --batch-size 8 \
    --lr 1e-4 \
    --unfreeze-blocks 4 \
    --workers 16

echo "=== Done: $(date) ==="

# ── Upload model to Oracle Object Storage ─────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ORACLE_ENDPOINT="https://idcsxwupyymi.compat.objectstorage.us-ashburn-1.oraclecloud.com"
DEST="s3://bloomi-training-data/trained-models/dinobloom_g_leukemia_classifier_${TIMESTAMP}.pth"

echo "=== Uploading model to Oracle Object Storage ==="
aws s3 cp $SCRATCH/dinobloom_g_finetuned.pth "$DEST" \
    --endpoint-url $ORACLE_ENDPOINT

echo "=== Model uploaded to: $DEST ==="

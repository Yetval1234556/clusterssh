#!/bin/bash
# ── Oracle A100 Training Script ────────────────────────────────────────────────
# Run directly on the Oracle A100 instance (no SLURM).
# Usage: bash train_oracle_a100.sh [num_gpus]
#   bash train_oracle_a100.sh 1   (single A100 80GB)
#   bash train_oracle_a100.sh 2   (if using VM.GPU.A100.2)
#
# ── Internet Fallback (tmux) ───────────────────────────────────────────────────
# Training runs inside a tmux session — if your Mac loses internet the training
# keeps running on the Oracle instance. Reconnect anytime with:
#   ssh opc@<instance-ip>
#   tmux attach -t dinobloom
#
# If tmux is not installed: sudo yum install -y tmux

TMUX_SESSION="dinobloom"

# If not already inside tmux, launch inside one and exit
if [ -z "$TMUX" ]; then
    echo "Launching training inside tmux session '$TMUX_SESSION'..."
    echo "Reconnect anytime with: tmux attach -t $TMUX_SESSION"
    tmux new-session -d -s $TMUX_SESSION "bash $0 $@"
    echo "Training running in background. Safe to close this terminal."
    exit 0
fi

set -e

NGPUS=${1:-1}
SCRATCH=$HOME/bloomi

# ── Environment ───────────────────────────────────────────────────────────────
source $HOME/.bashrc
# Source conda init from common locations (needed in some shells)
for f in "$HOME/miniconda3/etc/profile.d/conda.sh" \
          "$HOME/anaconda3/etc/profile.d/conda.sh" \
          "/opt/conda/etc/profile.d/conda.sh"; do
    [ -f "$f" ] && source "$f" && break
done
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

# Model uploads are handled automatically by the training script
# whenever test accuracy improves — no manual upload needed.

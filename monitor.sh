#!/bin/bash
# ── DinoBloom-G Remote Monitor ─────────────────────────────────────────────────
# Run this on your Mac/PC to monitor GPU training progress remotely.
# Automatically reconnects if your internet drops.
#
# Usage:
#   chmod +x monitor.sh
#   ./monitor.sh              — monitor H200 cluster
#   ./monitor.sh connect      — just SSH into cluster (no monitor)

# ── Config ─────────────────────────────────────────────────────────────────────
CLUSTER_HOST="login-01.ncshare.org"
CLUSTER_USER="rpatel1"
SCRATCH="/hpc/home/rpatel1/bloomi"
REFRESH=10
# ──────────────────────────────────────────────────────────────────────────────

MODE="monitor"
if [ "${1}" = "connect" ]; then
    MODE="connect"
fi

SSH_OPTS="-o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ConnectTimeout=10"

# ── Connect mode ───────────────────────────────────────────────────────────────
if [ "$MODE" = "connect" ]; then
    echo "Connecting to $CLUSTER_USER@$CLUSTER_HOST..."
    while true; do
        ssh $SSH_OPTS $CLUSTER_USER@$CLUSTER_HOST
        [ $? -eq 0 ] && break
        echo "Connection lost. Reconnecting in 5s... (Ctrl+C to stop)"
        sleep 5
    done
    exit 0
fi

# ── Monitor mode ───────────────────────────────────────────────────────────────
echo "========================================================"
echo "  DinoBloom-G Training Monitor"
echo "  Host    : $CLUSTER_USER@$CLUSTER_HOST"
echo "  Refresh : every ${REFRESH}s"
echo "  Ctrl+C  to stop"
echo "========================================================"
echo ""

while true; do
    clear
    echo "========================================================"
    echo "  DinoBloom-G — $(date)"
    echo "  $CLUSTER_USER@$CLUSTER_HOST"
    echo "========================================================"

    ssh $SSH_OPTS $CLUSTER_USER@$CLUSTER_HOST bash << REMOTE
        SCRATCH=$SCRATCH

        echo ""
        echo "--- GPU Status ---"
        nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.free,temperature.gpu,power.draw \
            --format=csv,noheader,nounits 2>/dev/null | \
            awk -F',' '{printf "  GPU: %-20s | Util: %s%% | Mem: %s/%s MB | Temp: %s°C | Power: %sW\n", \$1,\$2,\$4,(\$4+\$5),\$6,\$7}'

        echo ""
        echo "--- SLURM Jobs ---"
        squeue -u \$USER

        echo ""
        echo "--- Training Metrics ---"
        if [ -f "\$SCRATCH/training_metrics.csv" ]; then
            column -t -s',' "\$SCRATCH/training_metrics.csv"
            LAST=\$(tail -1 "\$SCRATCH/training_metrics.csv")
            EPOCH=\$(echo \$LAST | cut -d',' -f1)
            TRAIN=\$(echo \$LAST | cut -d',' -f3)
            TEST=\$(echo \$LAST | cut -d',' -f4)
            BEST=\$(echo \$LAST | cut -d',' -f6)
            echo "  Latest → Epoch: \$EPOCH | Train: \${TRAIN}% | Test: \${TEST}% | Best: \${BEST}%"
        else
            echo "  No metrics yet."
        fi

        echo ""
        echo "--- Recent Log ---"
        LATEST=\$(ls -t \$SCRATCH/logs/dino_*.out 2>/dev/null | head -1)
        if [ -n "\$LATEST" ]; then
            echo "  (\$LATEST)"
            tail -15 "\$LATEST"
        else
            echo "  No log file yet."
        fi
REMOTE

    if [ $? -ne 0 ]; then
        echo ""
        echo "Connection lost. Reconnecting in 5s... (Ctrl+C to stop)"
        sleep 5
        continue
    fi

    echo ""
    echo "Refreshing in ${REFRESH}s... (Ctrl+C to stop)"
    sleep $REFRESH
done

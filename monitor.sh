#!/bin/bash
# ── DinoBloom-G Remote Monitor ─────────────────────────────────────────────────
# Run this on your Mac to monitor GPU training progress remotely.
# Automatically reconnects if your internet drops.
#
# Usage:
#   chmod +x monitor.sh
#   ./monitor.sh unc              — monitor UNC H200 cluster
#   ./monitor.sh oracle           — monitor Oracle A100 instance
#   ./monitor.sh unc connect      — just SSH into UNC (no monitor)
#   ./monitor.sh oracle connect   — just SSH into Oracle

# ── Config — update these ──────────────────────────────────────────────────────
UNC_HOST="ncshare.login.org"
UNC_USER="rpatel1"
ORACLE_HOST="YOUR_ORACLE_INSTANCE_IP"
ORACLE_USER="opc"
REFRESH=10   # seconds between monitor refreshes
# ──────────────────────────────────────────────────────────────────────────────

TARGET=${1:-unc}

# Parse $2: "connect" means connect mode, anything else is a SLURM job ID
JOB_ID=""
MODE="monitor"
if [ "${2}" = "connect" ]; then
    MODE="connect"
elif [ -n "${2}" ]; then
    JOB_ID=${2}
fi

if [ "$TARGET" = "oracle" ]; then
    HOST=$ORACLE_HOST
    SSH_USER=$ORACLE_USER
    SCRATCH="\$HOME/bloomi"
else
    HOST=$UNC_HOST
    SSH_USER=$UNC_USER
    SCRATCH="/scratch/\$USER/bloomi"
fi

# ── SSH options for resilience ─────────────────────────────────────────────────
SSH_OPTS="-o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ConnectTimeout=10"

# ── Just connect mode ──────────────────────────────────────────────────────────
if [ "$MODE" = "connect" ]; then
    echo "Connecting to $SSH_USER@$HOST..."
    while true; do
        ssh $SSH_OPTS $SSH_USER@$HOST
        EXIT=$?
        [ $EXIT -eq 0 ] && break
        echo "Connection lost. Reconnecting in 5s... (Ctrl+C to stop)"
        sleep 5
    done
    exit 0
fi

# ── Monitor mode ───────────────────────────────────────────────────────────────
echo "========================================================"
echo "  DinoBloom-G Training Monitor"
echo "  Target  : $TARGET ($SSH_USER@$HOST)"
echo "  Refresh : every ${REFRESH}s"
echo "  Ctrl+C  to stop"
echo "========================================================"
echo ""

while true; do
    clear
    echo "========================================================"
    echo "  DinoBloom-G — $(date)"
    echo "  $TARGET | $SSH_USER@$HOST"
    echo "========================================================"

    ssh $SSH_OPTS $SSH_USER@$HOST bash << REMOTE
        SCRATCH=$SCRATCH

        echo ""
        echo "--- GPU Status ---"
        nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.free,temperature.gpu,power.draw \
            --format=csv,noheader,nounits 2>/dev/null | \
            awk -F',' '{printf "  GPU: %-20s | Util: %s%% | Mem: %s/%s MB | Temp: %s°C | Power: %sW\n", \$1,\$2,\$4,(\$4+\$5),\$6,\$7}'

        echo ""
        echo "--- Training Metrics ---"
        if [ -f "\$SCRATCH/training_metrics.csv" ]; then
            echo ""
            column -t -s',' "\$SCRATCH/training_metrics.csv"
            echo ""
            LAST=\$(tail -1 "\$SCRATCH/training_metrics.csv")
            EPOCH=\$(echo \$LAST | cut -d',' -f1)
            TRAIN=\$(echo \$LAST | cut -d',' -f3)
            TEST=\$(echo \$LAST | cut -d',' -f4)
            BEST=\$(echo \$LAST | cut -d',' -f6)
            TS=\$(echo \$LAST | cut -d',' -f7)
            echo "  Latest → Epoch: \$EPOCH | Train: \${TRAIN}% | Test: \${TEST}% | Best: \${BEST}% | \$TS"
        else
            echo "  No metrics yet — training may still be starting up."
        fi

        echo ""
        echo "--- Recent Log ---"
        LATEST=\$(ls -t \$SCRATCH/logs/dino_*.out \$SCRATCH/logs/*.out \$SCRATCH/logs/run_*.txt 2>/dev/null | head -1)
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

#!/bin/bash
# Remote commands piped via SSH by monitor.bat
# SCRATCH is exported by monitor.bat before calling this script

echo ""
echo "--- GPU Status ---"
nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.free,temperature.gpu,power.draw \
    --format=csv,noheader,nounits 2>/dev/null | \
    awk -F',' '{printf "  GPU: %-20s | Util: %s%% | Mem: %s/%s MB | Temp: %sC | Power: %sW\n", $1,$2,$4,($4+$5),$6,$7}'

echo ""
echo "--- SLURM Jobs ---"
squeue -u $USER

echo ""
echo "--- Training Metrics ---"
if [ -f "$SCRATCH/training_metrics.csv" ]; then
    column -t -s',' "$SCRATCH/training_metrics.csv"
    LAST=$(tail -1 "$SCRATCH/training_metrics.csv")
    EPOCH=$(echo $LAST | cut -d',' -f1)
    TRAIN=$(echo $LAST | cut -d',' -f3)
    TEST=$(echo $LAST | cut -d',' -f4)
    BEST=$(echo $LAST | cut -d',' -f6)
    echo "  Latest -> Epoch: $EPOCH | Train: ${TRAIN}% | Test: ${TEST}% | Best: ${BEST}%"
else
    echo "  No metrics yet."
fi

echo ""
echo "--- Recent Log ---"
LATEST=$(ls -t $SCRATCH/logs/dino_*.out 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
    echo "  ($LATEST)"
    tail -15 "$LATEST"
else
    echo "  No log file yet."
fi

#!/bin/bash
# ── Live Training Monitor ──────────────────────────────────────────────────────
# Run this on the cluster while training is running to see live progress.
# Usage: bash watch_training.sh [job_id]

SCRATCH=/scratch/$USER/bloomi
METRICS=$SCRATCH/training_metrics.csv
LOGS_DIR=$SCRATCH/logs

echo "=== DinoBloom-G Training Monitor ==="
echo "Scratch : $SCRATCH"
echo ""

# Show job status
echo "--- SLURM Job Status ---"
squeue -u $USER -o "%.10i %.9P %.12j %.8T %.10M %.6D %R"
echo ""

# Show latest metrics from CSV
if [ -f "$METRICS" ]; then
    echo "--- Training Metrics ---"
    echo ""
    column -t -s',' "$METRICS"
    echo ""
    LAST=$(tail -1 "$METRICS")
    EPOCH=$(echo $LAST | cut -d',' -f1)
    TRAIN_ACC=$(echo $LAST | cut -d',' -f3)
    TEST_ACC=$(echo $LAST | cut -d',' -f4)
    BEST=$(echo $LAST | cut -d',' -f6)
    TIMESTAMP=$(echo $LAST | cut -d',' -f7)
    echo "--- Latest: Epoch $EPOCH | Train: ${TRAIN_ACC}% | Test: ${TEST_ACC}% | Best: ${BEST}% | $TIMESTAMP ---"
else
    echo "No metrics file yet — training may not have started."
fi

echo ""

# Show latest GPU monitor stats
JOB_ID=${1:-""}
if [ -n "$JOB_ID" ] && [ -f "$LOGS_DIR/gpu_monitor_${JOB_ID}.csv" ]; then
    echo "--- Latest GPU Stats ---"
    tail -9 "$LOGS_DIR/gpu_monitor_${JOB_ID}.csv" | column -t -s','
elif [ -f "$(ls -t $LOGS_DIR/gpu_monitor_*.csv 2>/dev/null | head -1)" ]; then
    LATEST_GPU=$(ls -t $LOGS_DIR/gpu_monitor_*.csv | head -1)
    echo "--- Latest GPU Stats ($LATEST_GPU) ---"
    tail -9 "$LATEST_GPU" | column -t -s','
else
    echo "No GPU monitor log yet. Pass job ID: bash watch_training.sh <job_id>"
fi

echo ""
echo "--- Recent Log Output ---"
LATEST_LOG=$(ls -t $LOGS_DIR/run_*.txt 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "(from $LATEST_LOG)"
    tail -30 "$LATEST_LOG"
else
    echo "No log file found yet."
fi

#!/bin/bash
# ── Connect to UNC Cluster or Oracle Instance ──────────────────────────────────
# Usage:
#   ./connect.sh           (UNC cluster)
#   ./connect.sh oracle    (Oracle A100 instance)
#
# Auto-reconnects if connection drops — safe to run on flaky internet.

UNC_HOST="YOUR_CLUSTER.unc.edu"
ORACLE_HOST="YOUR_ORACLE_INSTANCE_IP"
CLUSTER_USER="$USER"

TARGET=${1:-unc}

if [ "$TARGET" = "oracle" ]; then
    HOST=$ORACLE_HOST
    USER_AT="opc"
else
    HOST=$UNC_HOST
    USER_AT=$CLUSTER_USER
fi

echo "Connecting to $USER_AT@$HOST (auto-reconnect enabled)..."
echo "Press Ctrl+C to stop reconnecting."
echo ""

# Auto-reconnect loop — retries every 5 seconds if connection drops
while true; do
    ssh -o ServerAliveInterval=60 \
        -o ServerAliveCountMax=3 \
        -o ExitOnForwardFailure=yes \
        $USER_AT@$HOST
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Session ended normally."
        break
    fi
    echo "Connection lost (exit $EXIT_CODE). Reconnecting in 5 seconds... (Ctrl+C to stop)"
    sleep 5
done

#!/bin/bash
# ── Connect to UNC Cluster ─────────────────────────────────────────────────────
# Usage: bash connect.sh
# Make executable: chmod +x connect.sh

CLUSTER_HOST="YOUR_CLUSTER.unc.edu"
CLUSTER_USER="$USER"

echo "Connecting to $CLUSTER_USER@$CLUSTER_HOST..."
ssh $CLUSTER_USER@$CLUSTER_HOST

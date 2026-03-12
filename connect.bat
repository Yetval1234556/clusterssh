@echo off
REM ── Connect to UNC Cluster ─────────────────────────────────────────────────
REM Update CLUSTER_HOST with your cluster's address before running

set CLUSTER_HOST=YOUR_CLUSTER.unc.edu
set CLUSTER_USER=%USERNAME%

echo Connecting to %CLUSTER_HOST%...
ssh %CLUSTER_USER%@%CLUSTER_HOST%
pause

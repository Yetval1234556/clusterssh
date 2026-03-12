#!/bin/bash
# ── UNC Cluster Setup & Training Script ───────────────────────────────────────
# Usage: bash setup.sh
# Run this once after SSHing into the cluster to set up and launch training.

CLUSTER_USER=$USER
SCRATCH=/scratch/$CLUSTER_USER
REPO_URL="https://github.com/Yetval1234556/DinoModelsEXTRA"
ORACLE_ENDPOINT="https://idcsxwupyymi.compat.objectstorage.us-ashburn-1.oraclecloud.com"
ORACLE_BUCKET="bloomi-training-data"
AWS_ACCESS_KEY="1b37303fea13e610d55230c34d828d39ec85c7af"
AWS_SECRET_KEY="50Patf4/wPWycN35HGK17JboDTHd+3T0/90nfQNU3x0="

echo "=== Setting up DinoBloom-G training on cluster ==="

# 1. Configure Oracle S3 credentials
echo "[1/6] Configuring Oracle Object Storage credentials..."
aws configure set aws_access_key_id $AWS_ACCESS_KEY
aws configure set aws_secret_access_key $AWS_SECRET_KEY
aws configure set region us-ashburn-1

# 2. Clone the repo
echo "[2/6] Cloning repository..."
cd $SCRATCH
git clone $REPO_URL bloomi
cd bloomi

# 3. Pull dataset from Oracle Object Storage
echo "[3/6] Downloading dataset from Oracle Object Storage (~10GB)..."
mkdir -p $SCRATCH/bloomi/New\ Data/extracted
aws s3 cp s3://$ORACLE_BUCKET/extracted/ "$SCRATCH/bloomi/New Data/extracted/" \
    --recursive \
    --endpoint-url $ORACLE_ENDPOINT

# 4. Download DinoBloom-G pretrained weights
echo "[4/6] Downloading DinoBloom-G pretrained weights (~4.4GB)..."
# Place your DinoBloom-G.pth download command here
# e.g. wget or aws s3 cp if uploaded to Oracle
# wget -O DinoBloom-G.pth <URL>

# 5. Set up conda environment
echo "[5/6] Setting up conda environment..."
module load conda 2>/dev/null || true
conda env create -f conda.yaml -n dinov2 || conda env update -f conda.yaml -n dinov2
conda activate dinov2

# 6. Submit training job
echo "[6/6] Submitting SLURM job..."
sbatch train_cluster.sh

echo "=== Done! Check job status with: squeue -u $USER ==="

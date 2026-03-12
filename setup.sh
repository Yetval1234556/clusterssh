#!/bin/bash
# ── UNC Cluster Setup Script ───────────────────────────────────────────────────
# Usage: bash setup.sh
# Run this once after SSHing into the cluster to set up the environment.

set -e

CLUSTER_USER=$USER
SCRATCH=/scratch/$CLUSTER_USER
REPO_URL="https://github.com/Yetval1234556/DinoModelsEXTRA"
ORACLE_BUCKET="bloomi-training-data"
ORACLE_NAMESPACE="idcsxwupyymi"
ORACLE_REGION="us-ashburn-1"
OCI_USER="ocid1.user.oc1..aaaaaaaa62h4eh56dbwmetzvjnqmhpc6to4horuxtfwmauhjk6dqckzhkjza"
OCI_TENANCY="ocid1.tenancy.oc1..aaaaaaaaxz235iw5sjl4jhzu7xuo6rcasflfxyrrx4h6murstpvh6c6chlfq"
OCI_FINGERPRINT="4d:5e:8f:86:13:55:c7:83:4b:04:2e:3e:9b:1a:2c:c9"

echo "=== Setting up DinoBloom-G training on cluster ==="
echo "User    : $CLUSTER_USER"
echo "Scratch : $SCRATCH"
echo ""

# 1. Configure OCI CLI
echo "[1/6] Configuring OCI CLI..."
mkdir -p ~/.oci
cat > ~/.oci/config << EOF
[DEFAULT]
user=${OCI_USER}
fingerprint=${OCI_FINGERPRINT}
tenancy=${OCI_TENANCY}
region=${ORACLE_REGION}
EOF
chmod 600 ~/.oci/config
oci os ns get > /dev/null && echo "  OCI CLI connected OK" || echo "  WARNING: OCI CLI auth failed"

# 2. Clone the repo
echo "[2/6] Cloning repository..."
mkdir -p $SCRATCH
cd $SCRATCH
if [ -d "bloomi/.git" ]; then
    echo "  Repo exists, updating..."
    cd bloomi && git pull
else
    git clone $REPO_URL bloomi
    cd bloomi
fi

# 3. Pull dataset from Oracle Object Storage using OCI CLI
echo "[3/6] Downloading dataset from Oracle Object Storage (~10GB)..."
mkdir -p "$SCRATCH/bloomi/New Data/extracted"
oci os object bulk-download \
    --namespace $ORACLE_NAMESPACE \
    --bucket-name $ORACLE_BUCKET \
    --download-dir "$SCRATCH/bloomi/New Data/extracted" \
    --prefix "extracted/" \
    --overwrite
echo "  Dataset downloaded."

# 4. Download DinoBloom-G pretrained weights
echo "[4/6] Downloading DinoBloom-G pretrained weights (~4.4GB)..."
oci os object get \
    --namespace $ORACLE_NAMESPACE \
    --bucket-name $ORACLE_BUCKET \
    --name "DinoBloom-G.pth" \
    --file "$SCRATCH/bloomi/DinoBloom-G.pth"
echo "  Weights downloaded."

# 5. Set up conda environment
echo "[5/6] Setting up conda environment..."
module load conda 2>/dev/null || true
# Source conda init from common locations (needed in non-interactive shells)
for f in "$HOME/miniconda3/etc/profile.d/conda.sh" \
          "$HOME/anaconda3/etc/profile.d/conda.sh" \
          "/opt/conda/etc/profile.d/conda.sh"; do
    [ -f "$f" ] && source "$f" && break
done
conda env create -f conda.yaml -n dinov2 2>/dev/null || conda env update -f conda.yaml -n dinov2
echo "  Conda env ready."

# 6. Ready to submit
echo "[6/6] Setup complete — submit your training job:"
echo "  sbatch train_unc_h200.sh       (2x H200, default)"
echo "  sbatch train_unc_h200.sh 1     (single H200)"
echo ""
echo "Monitor with: squeue -u $USER"
echo "View logs  : tail -f $SCRATCH/bloomi/logs/dino_<jobid>.out"

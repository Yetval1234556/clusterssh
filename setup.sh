#!/bin/bash
# ── ncshare H200 Cluster Setup Script ─────────────────────────────────────────
# Run this once on the cluster to set up the environment.
# Usage: bash setup.sh

# Strip Windows line endings if present
sed -i 's/\r//' "$0" 2>/dev/null || true

set -e

SCRATCH=/hpc/home/$USER/bloomi
REPO_URL="https://github.com/Yetval1234556/DinoModelsEXTRA.git"
ORACLE_BUCKET="bloomi-training-data"
ORACLE_NAMESPACE="idcsxwupyymi"
ORACLE_REGION="us-ashburn-1"
OCI_USER="ocid1.user.oc1..aaaaaaaa62h4eh56dbwmetzvjnqmhpc6to4horuxtfwmauhjk6dqckzhkjza"
OCI_TENANCY="ocid1.tenancy.oc1..aaaaaaaaxz235iw5sjl4jhzu7xuo6rcasflfxyrrx4h6murstpvh6c6chlfq"
OCI_FINGERPRINT="4d:5e:8f:86:13:55:c7:83:4b:04:2e:3e:9b:1a:2c:c9"

echo "=== Setting up DinoBloom-G on ncshare H200 ==="
echo "User    : $USER"
echo "Dir     : $SCRATCH"
echo ""

# 1. Configure OCI CLI
echo "[1/5] Configuring OCI CLI..."
export PATH="$HOME/.local/bin:$HOME/bin:/usr/local/bin:$PATH"
# If still not found, try to locate it
if ! command -v oci &>/dev/null; then
    OCI_BIN=$(find $HOME -name "oci" -type f 2>/dev/null | head -1)
    [ -n "$OCI_BIN" ] && export PATH="$(dirname $OCI_BIN):$PATH"
fi
mkdir -p ~/.oci
cat > ~/.oci/config << OCIEOF
[DEFAULT]
user=${OCI_USER}
fingerprint=${OCI_FINGERPRINT}
tenancy=${OCI_TENANCY}
region=${ORACLE_REGION}
OCIEOF
chmod 600 ~/.oci/config
oci os ns get > /dev/null && echo "  OCI CLI connected OK" || echo "  WARNING: OCI CLI auth failed — data download may fail"

# 2. Clone the repo (public, no auth needed)
echo "[2/5] Cloning repository..."
mkdir -p $SCRATCH
cd /hpc/home/$USER
if [ -d "bloomi/.git" ]; then
    echo "  Repo exists, updating..."
    cd bloomi && GIT_TERMINAL_PROMPT=0 git pull
else
    GIT_TERMINAL_PROMPT=0 git clone $REPO_URL bloomi
    cd bloomi
fi

# 3. Pull dataset from Oracle Object Storage
echo "[3/5] Downloading dataset from Oracle bucket (~10GB)..."
mkdir -p "$SCRATCH/New Data/extracted"
oci os object bulk-download \
    --namespace $ORACLE_NAMESPACE \
    --bucket-name $ORACLE_BUCKET \
    --download-dir "$SCRATCH/New Data/extracted" \
    --prefix "extracted/" \
    --overwrite
echo "  Dataset downloaded."

# 4. Download DinoBloom-G pretrained weights
echo "[4/5] Downloading DinoBloom-G weights (~4.4GB)..."
oci os object get \
    --namespace $ORACLE_NAMESPACE \
    --bucket-name $ORACLE_BUCKET \
    --name "DinoBloom-G.pth" \
    --file "$SCRATCH/DinoBloom-G.pth"
echo "  Weights downloaded."

# 5. Set up conda environment
echo "[5/5] Setting up conda environment..."
module load conda 2>/dev/null || true
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
fi
conda env create -f $SCRATCH/conda.yaml -n dinov2 2>/dev/null || conda env update -f $SCRATCH/conda.yaml -n dinov2
echo "  Conda env ready."

echo ""
echo "=== Setup complete — submit your training job: ==="
echo "  sbatch train_h200.sh       (2x H200, default)"
echo "  sbatch train_h200.sh 1     (1x H200)"
echo ""
echo "Monitor with: squeue -u $USER"

#!/bin/bash
# ── Oracle A100 Setup Script ───────────────────────────────────────────────────
# Run this once on the Oracle VM to set up the environment.
# Usage: bash setup.sh

set -e

SCRATCH=$HOME/bloomi
REPO_URL="https://github.com/Yetval1234556/DinoModelsEXTRA"
ORACLE_BUCKET="bloomi-training-data"
ORACLE_NAMESPACE="idcsxwupyymi"
ORACLE_REGION="us-ashburn-1"
OCI_USER="ocid1.user.oc1..aaaaaaaa62h4eh56dbwmetzvjnqmhpc6to4horuxtfwmauhjk6dqckzhkjza"
OCI_TENANCY="ocid1.tenancy.oc1..aaaaaaaaxz235iw5sjl4jhzu7xuo6rcasflfxyrrx4h6murstpvh6c6chlfq"
OCI_FINGERPRINT="4d:5e:8f:86:13:55:c7:83:4b:04:2e:3e:9b:1a:2c:c9"

echo "=== Setting up DinoBloom-G on Oracle A100 ==="
echo "Home : $HOME"
echo "Dir  : $SCRATCH"
echo ""

# 1. Configure OCI CLI
echo "[1/6] Configuring OCI CLI..."
pip install oci-cli --quiet 2>/dev/null || true
mkdir -p ~/.oci
cat > ~/.oci/config << 'OCIEOF'
[DEFAULT]
OCIEOF
cat >> ~/.oci/config << OCIEOF
user=${OCI_USER}
fingerprint=${OCI_FINGERPRINT}
tenancy=${OCI_TENANCY}
region=${ORACLE_REGION}
OCIEOF
chmod 600 ~/.oci/config
oci os ns get > /dev/null && echo "  OCI CLI connected OK" || echo "  WARNING: OCI CLI auth failed"

# 2. Clone the repo
echo "[2/6] Cloning repository..."
mkdir -p $SCRATCH
cd $HOME
if [ -d "bloomi/.git" ]; then
    echo "  Repo exists, updating..."
    cd bloomi && git pull
else
    git clone $REPO_URL bloomi
    cd bloomi
fi

# 3. Pull dataset from Oracle Object Storage
echo "[3/6] Downloading dataset (~10GB)..."
mkdir -p "$SCRATCH/New Data/extracted"
oci os object bulk-download \
    --namespace $ORACLE_NAMESPACE \
    --bucket-name $ORACLE_BUCKET \
    --download-dir "$SCRATCH/New Data/extracted" \
    --prefix "extracted/" \
    --overwrite
echo "  Dataset downloaded."

# 4. Download DinoBloom-G pretrained weights
echo "[4/6] Downloading DinoBloom-G weights (~4.4GB)..."
oci os object get \
    --namespace $ORACLE_NAMESPACE \
    --bucket-name $ORACLE_BUCKET \
    --name "DinoBloom-G.pth" \
    --file "$SCRATCH/DinoBloom-G.pth"
echo "  Weights downloaded."

# 5. Set up conda environment
echo "[5/6] Setting up conda environment..."
if ! command -v conda &>/dev/null; then
    echo "  Installing Miniconda..."
    curl -sLo /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
fi
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null || \
eval "$(conda shell.bash hook)"
conda env create -f $SCRATCH/conda.yaml -n dinov2 2>/dev/null || \
conda env update -f $SCRATCH/conda.yaml -n dinov2
echo "  Conda env ready."

# 6. Done
echo "[6/6] Setup complete — start training:"
echo "  bash train_oracle_a100.sh 1    (1x A100)"
echo "  bash train_oracle_a100.sh 2    (2x A100)"
echo ""
echo "Monitor from your PC: monitor.bat oracle"

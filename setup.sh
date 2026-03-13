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

# 1. Locate or install OCI CLI
echo "[1/5] Locating OCI CLI..."
echo "  Checking standard PATH locations..."
export PATH="$HOME/.local/bin:$HOME/bin:/usr/local/bin:$PATH"

# Step 1a: try module load (HPC clusters often ship oci-cli as a module)
if ! command -v oci &>/dev/null; then
    echo "  Trying: module load oci-cli..."
    module load oci-cli 2>/dev/null && echo "  Loaded via module system." || echo "  Module not available."
fi

# Step 1b: try known install locations
if ! command -v oci &>/dev/null; then
    echo "  Checking known install directories..."
    for dir in \
        "$HOME/.local/bin" \
        "$HOME/bin" \
        "$HOME/lib/oracle-cli/bin" \
        "$HOME/.oci/bin" \
        "/usr/local/bin" \
        "/opt/oci-cli/bin" \
        "/opt/oracle/oci/bin"; do
        if [ -f "$dir/oci" ]; then
            export PATH="$dir:$PATH"
            echo "  Found OCI CLI at: $dir/oci"
            break
        fi
    done
fi

# Step 1c: broad search in $HOME
if ! command -v oci &>/dev/null; then
    echo "  Running broad search for 'oci' binary under \$HOME (may take a moment)..."
    OCI_BIN=$(find "$HOME" -name "oci" -type f 2>/dev/null | head -1)
    if [ -n "$OCI_BIN" ]; then
        export PATH="$(dirname $OCI_BIN):$PATH"
        echo "  Found OCI CLI at: $OCI_BIN"
    else
        echo "  OCI CLI not found anywhere on this system."
    fi
fi

# Step 1d: pip install as last resort
if ! command -v oci &>/dev/null; then
    echo ""
    echo "  !! OCI CLI missing — installing via pip (this takes 2-4 minutes)..."
    echo "  Command: pip install --user oci-cli"
    echo "  ────────────────────────────────────────────────────────────────"
    pip install --user oci-cli \
        --progress-bar on \
        2>&1 | while IFS= read -r line; do echo "  pip | $line"; done
    export PATH="$HOME/.local/bin:$PATH"
    echo "  ────────────────────────────────────────────────────────────────"
    if command -v oci &>/dev/null; then
        echo "  pip install succeeded."
    else
        echo "  FATAL: pip install finished but 'oci' still not found."
        echo "  Try manually: pip install oci-cli && export PATH=\$HOME/.local/bin:\$PATH"
        exit 1
    fi
fi

echo ""
echo "  OCI CLI binary  : $(which oci)"
echo "  OCI CLI version : $(oci --version 2>&1)"
echo ""

# Step 1e: write config
echo "  Writing ~/.oci/config..."
mkdir -p ~/.oci
cat > ~/.oci/config << OCIEOF
[DEFAULT]
user=${OCI_USER}
fingerprint=${OCI_FINGERPRINT}
tenancy=${OCI_TENANCY}
region=${ORACLE_REGION}
OCIEOF
chmod 600 ~/.oci/config
echo "  Config written to ~/.oci/config"

echo "  Testing OCI auth..."
if oci os ns get > /dev/null 2>&1; then
    echo "  OCI CLI auth   : OK (namespace=$(oci os ns get --query data --raw-output 2>/dev/null))"
else
    echo "  WARNING: OCI CLI auth test failed — check credentials or network."
    echo "  Data download steps below may fail."
fi

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
echo "  sbatch train_h200.sh       (4x H200, default)"
echo "  sbatch train_h200.sh 2     (2x H200)"
echo "  sbatch train_h200.sh 1     (1x H200)"
echo ""
echo "Monitor with: squeue -u $USER"

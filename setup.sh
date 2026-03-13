#!/bin/bash
# ── ncshare H200 Cluster Setup Script ─────────────────────────────────────────
# Run this once on the cluster to set up the environment.
# Usage: bash setup.sh

# Strip Windows line endings if present
sed -i 's/\r//' "$0" 2>/dev/null || true

set -e

SCRATCH=/hpc/home/$USER/bloomi
ORACLE_BUCKET="bloomi-training-data"
ORACLE_NAMESPACE="idcsxwupyymi"
ORACLE_REGION="us-ashburn-1"
OCI_USER="ocid1.user.oc1..aaaaaaaa62h4eh56dbwmetzvjnqmhpc6to4horuxtfwmauhjk6dqckzhkjza"
OCI_TENANCY="ocid1.tenancy.oc1..aaaaaaaaxz235iw5sjl4jhzu7xuo6rcasflfxyrrx4h6murstpvh6c6chlfq"
OCI_FINGERPRINT="e5:16:5f:27:13:84:76:c0:31:f3:88:ef:28:2c:32:08"

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

# Step 1d: install as last resort — try pipx first, then pip --break-system-packages
if ! command -v oci &>/dev/null; then
    echo ""
    echo "  !! OCI CLI missing — attempting install (this takes 2-4 minutes)..."
    echo "  ────────────────────────────────────────────────────────────────"

    # Try pipx first (cleanest — manages its own venv, no system interference)
    if command -v pipx &>/dev/null; then
        echo "  Method: pipx install oci-cli"
        pipx install oci-cli 2>&1 | while IFS= read -r line; do echo "  pipx | $line"; done
        export PATH="$HOME/.local/bin:$PATH"
    fi

    # Try pip --user (standard)
    if ! command -v oci &>/dev/null; then
        echo "  Method: pip install --user oci-cli"
        pip install --user oci-cli --progress-bar on \
            2>&1 | while IFS= read -r line; do echo "  pip  | $line"; done
        export PATH="$HOME/.local/bin:$PATH"
    fi

    # Try pip --user --break-system-packages (Debian/Ubuntu PEP 668 systems)
    if ! command -v oci &>/dev/null; then
        echo "  System Python is externally managed (PEP 668 / Debian)."
        echo "  Method: pip install --user --break-system-packages oci-cli"
        pip install --user --break-system-packages oci-cli --progress-bar on \
            2>&1 | while IFS= read -r line; do echo "  pip  | $line"; done
        export PATH="$HOME/.local/bin:$PATH"
    fi

    echo "  ────────────────────────────────────────────────────────────────"
    if command -v oci &>/dev/null; then
        echo "  Install succeeded."
    else
        echo "  FATAL: could not install OCI CLI via pipx or pip."
        echo "  Ask your sysadmin to install oci-cli, or run:"
        echo "    pip install --user --break-system-packages oci-cli"
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
key_file=~/.oci/oci_api_key.pem
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

# 2. Create working directory (training scripts already SCPd by setup.bat)
echo "[2/5] Preparing working directory..."
mkdir -p $SCRATCH
echo "  Scratch dir : $SCRATCH"
echo "  Training scripts expected in $SCRATCH (SCPd by setup.bat)"
if [ -f "$SCRATCH/train_efficientnet_b0.py" ] && [ -f "$SCRATCH/train_efficientnet_b0_ddp.py" ]; then
    echo "  train_efficientnet_b0.py     : OK"
    echo "  train_efficientnet_b0_ddp.py : OK"
else
    echo "  WARNING: Training scripts not found in $SCRATCH"
    echo "  Make sure setup.bat completed successfully before running the training job."
fi

# 3. Pull dataset from Oracle Object Storage
echo "[3/5] Checking dataset..."
mkdir -p "$SCRATCH/New Data/extracted"
DATASET_COUNT=$(find "$SCRATCH/New Data/extracted" -name "*.jpg" 2>/dev/null | wc -l)
if [ "$DATASET_COUNT" -gt 1000 ]; then
    echo "  Dataset already present ($DATASET_COUNT images) — skipping download."
else
    echo "  Downloading dataset from Oracle (~10GB)..."
    oci os object bulk-download \
        --namespace $ORACLE_NAMESPACE \
        --bucket-name $ORACLE_BUCKET \
        --download-dir "$SCRATCH/New Data/extracted" \
        --prefix "extracted/" \
        --overwrite
    echo "  Dataset downloaded."
fi

# 4. Download DinoBloom-G pretrained weights
echo "[4/5] Checking DinoBloom-G weights..."
WEIGHTS_SIZE=$(stat -c%s "$SCRATCH/DinoBloom-G.pth" 2>/dev/null || echo 0)
if [ "$WEIGHTS_SIZE" -gt 104857600 ]; then  # must be >100MB to be valid
    echo "  Weights already present ($(du -sh "$SCRATCH/DinoBloom-G.pth" | cut -f1)) — skipping download."
else
    [ "$WEIGHTS_SIZE" -gt 0 ] && echo "  Existing weights file is too small ($WEIGHTS_SIZE bytes) — redownloading..."
    echo "  Downloading DinoBloom-G weights (~4.4GB)..."
    rm -f "$SCRATCH/DinoBloom-G.pth"
    oci os object get \
        --namespace $ORACLE_NAMESPACE \
        --bucket-name $ORACLE_BUCKET \
        --name "trained-models/dinobloom/dinobloom_g_finetuned.pth" \
        --file "$SCRATCH/DinoBloom-G.pth"
    echo "  Weights downloaded: $(du -sh "$SCRATCH/DinoBloom-G.pth" | cut -f1)"
fi

# 5. Set up conda environment
echo "[5/5] Checking conda environment..."

# Source .bashrc first — cluster may set up conda there
source "$HOME/.bashrc" 2>/dev/null || true

# Try every common module name
for mod in conda anaconda anaconda3 miniconda miniconda3 python/anaconda python/miniconda anaconda/latest miniconda/latest; do
    command -v conda &>/dev/null && break
    module load $mod 2>/dev/null || true
done

# Source conda.sh from known locations
for f in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "$HOME/miniconda/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh" \
    "/usr/local/conda/etc/profile.d/conda.sh" \
    "/hpc/software/anaconda3/etc/profile.d/conda.sh" \
    "/hpc/software/miniconda3/etc/profile.d/conda.sh" \
    "/apps/anaconda3/etc/profile.d/conda.sh"; do
    [ -f "$f" ] && source "$f" && echo "  Sourced conda from: $f" && break
done

# Broad search as last resort
if ! command -v conda &>/dev/null; then
    echo "  Searching for conda binary..."
    CONDA_BIN=$(find /hpc /apps /software /opt "$HOME" -name "conda" -type f 2>/dev/null | grep -v __pycache__ | head -1)
    if [ -n "$CONDA_BIN" ]; then
        export PATH="$(dirname $CONDA_BIN):$PATH"
        echo "  Found conda at: $CONDA_BIN"
        CONDA_SH="$(dirname $CONDA_BIN)/../etc/profile.d/conda.sh"
        [ -f "$CONDA_SH" ] && source "$CONDA_SH"
    fi
fi

if ! command -v conda &>/dev/null; then
    echo "  FATAL: conda not found. Run 'module avail' on the cluster to find the correct module name."
    exit 1
fi

echo "  Conda binary: $(which conda)  version: $(conda --version)"

if conda env list | grep -q "^dinov2 "; then
    echo "  Conda env 'dinov2' already exists — skipping creation."
    conda env update -f $SCRATCH/conda.yaml -n dinov2 --prune 2>/dev/null || true
else
    echo "  Creating conda env 'dinov2'..."
    conda env create -f $SCRATCH/conda.yaml -n dinov2
fi
echo "  Conda env ready."

echo ""
echo "=== Setup complete — submit your training job: ==="
echo "  sbatch train_h200.sh       (4x H200, default)"
echo "  sbatch train_h200.sh 2     (2x H200)"
echo "  sbatch train_h200.sh 1     (1x H200)"
echo ""
echo "Monitor with: squeue -u $USER"

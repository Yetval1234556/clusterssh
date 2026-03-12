# Cluster SSH — DinoBloom-G Training Setup

Scripts for connecting to GPU infrastructure and launching DinoBloom-G leukemia classifier fine-tuning. Supports two environments: UNC H200 cluster (SLURM) and Oracle Cloud A100 VM.

## Files

| File | Description |
|------|-------------|
| `monitor.sh` | Mac-side monitor — GPU stats, metrics, logs, auto-reconnects if internet drops |
| `setup.sh` | One-time setup on cluster — pulls data, clones repo, sets up conda |
| `train_unc_h200.sh` | SLURM job for UNC H200 cluster (8x H200, 96GB VRAM each) |
| `train_oracle_a100.sh` | Direct training script for Oracle A100 VM (80GB VRAM, tmux fallback) |

Multi-GPU training requires `train_efficientnet_b0_ddp.py` from [DinoModelsEXTRA](https://github.com/Yetval1234556/DinoModelsEXTRA).

---

## UNC H200 Cluster

### 1. Connect & Monitor
Edit `monitor.sh`, fill in `UNC_HOST` and `ORACLE_HOST`, then from Mac Terminal:
```bash
chmod +x monitor.sh

./monitor.sh unc connect      # SSH into UNC cluster
./monitor.sh oracle connect   # SSH into Oracle instance
./monitor.sh unc              # Live GPU monitor (UNC)
./monitor.sh oracle           # Live GPU monitor (Oracle)
```
Auto-reconnects every 5 seconds if your Mac loses internet.

### 2. One-time setup
```bash
bash setup.sh
```
This will:
- Configure OCI CLI and Oracle S3 credentials (hardcoded — no manual key setup needed)
- Clone the DinoModelsEXTRA repo to scratch storage
- Download the 10GB dataset from Oracle Object Storage
- Set up the `dinov2` conda environment

### 3. Submit training job
```bash
# 8x H200 (recommended — effective batch size 512)
sbatch train_unc_h200.sh

# Single H200
sbatch train_unc_h200.sh 1
```

### 4. Monitor (from your Mac)
```bash
./monitor.sh unc
```
Shows GPU utilization, training metrics, and recent log output. Refreshes every 10s and auto-reconnects if internet drops.

### Cluster Specs
| Setting | Value |
|---------|-------|
| GPUs | 8x NVIDIA H200 (96GB VRAM each) |
| Partition | `gpu_p` |
| Constraint | `h200` |
| CPUs | 112 |
| RAM | 1800GB |
| Max time | 96 hours |
| Batch size | 64 per GPU (512 effective with 8 GPUs) |
| Epochs | 75 |

---

## Oracle A100 VM

### 1. Spin up instance
Oracle Console → Compute → Instances → Create Instance → Shape: `VM.GPU.A100.1`

### 2. Configure storage
When creating the instance set the boot volume to **500GB+** — Oracle gives unlimited storage the first month so use it.

### 3. SSH in and set up
```bash
ssh opc@<instance-ip>

# Clone repo
git clone https://github.com/Yetval1234556/DinoModelsEXTRA bloomi
cd bloomi

# Pull dataset from Oracle Object Storage using OCI CLI (credentials hardcoded)
oci os object bulk-download \
  --namespace idcsxwupyymi \
  --bucket-name bloomi-training-data \
  --download-dir "New Data/extracted" \
  --prefix "extracted/" \
  --overwrite

# Pull DinoBloom-G pretrained weights
oci os object get \
  --namespace idcsxwupyymi \
  --bucket-name bloomi-training-data \
  --name "DinoBloom-G.pth" \
  --file DinoBloom-G.pth

# Set up conda
conda env create -f conda.yaml -n dinov2
conda activate dinov2
```

### 4. Internet fallback
Training on Oracle runs inside a **tmux session** automatically. If your Mac loses internet the training keeps running. Reconnect anytime:
```bash
./monitor.sh oracle connect
tmux attach -t dinobloom
```

### 5. Run training
```bash
# Single A100 80GB
bash train_oracle_a100.sh 1

# 2x A100 (if using VM.GPU.A100.2)
bash train_oracle_a100.sh 2
```

### Oracle Specs
| Setting | Value |
|---------|-------|
| GPU | 1x NVIDIA A100 (80GB VRAM) |
| Storage | Unlimited first month — set boot volume to 500GB+ |
| Batch size | 64 per GPU |
| Epochs | 75 |
| Est. time | ~25-30 hours |
| Est. cost | ~$75-90 of $300 credits |

---

## Retrieving the Trained Model

After training completes the model is **automatically uploaded** to Oracle Object Storage:

```
bloomi-training-data/trained-models/dinobloom_g_h200_YYYYMMDD_HHMMSS.pth   ← UNC run
bloomi-training-data/trained-models/dinobloom_g_a100_YYYYMMDD_HHMMSS.pth   ← Oracle run
```

- Each run gets a unique timestamp — nothing is ever overwritten
- Browse in [Oracle Cloud Console](https://cloud.oracle.com) → Object Storage → `bloomi-training-data` → `trained-models/`

Download to your PC:
```bash
oci os object get \
  --namespace idcsxwupyymi \
  --bucket-name bloomi-training-data \
  --name "trained-models/<filename>.pth" \
  --file ~/Downloads/dinobloom_g_finetuned.pth
```

---

## Requirements
- OCI CLI installed (`pip install oci-cli`) — credentials are hardcoded in `setup.sh`
- AWS CLI installed — S3 credentials are hardcoded in `setup.sh`
- Conda with `dinov2` environment (created by `setup.sh`)
- CUDA 11.7+
- SLURM (UNC only)

## Notes
- `DinoBloom-G.pth` pretrained weights (~4.4GB) must be present in the repo directory before training
- Dataset is stored in Oracle Object Storage bucket `bloomi-training-data`
- Training metrics saved to `training_metrics.csv` every epoch
- GPU stats logged every 30s to `logs/gpu_monitor_<id>.csv`

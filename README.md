# clusterssh

Scripts for running DinoBloom-G leukemia classifier fine-tuning on the ncshare H200 cluster.
**Windows only** for local/GitHub use (`.bat` files). Cluster runs Linux (`.sh` files).

Cluster: `rpatel1@login-01.ncshare.org`
Storage: Oracle Object Storage (`bloomi-training-data`, region `us-ashburn-1`)

---

## Files

| File | Where it runs | What it does |
|------|--------------|--------------|
| `setup.bat` | Windows (local) | One-stop setup: caches SSH passphrase, SCPs everything to cluster, runs remote setup |
| `train_h200.bat` | Windows (local) | SCPs latest training code to cluster, submits `sbatch` job |
| `setup.sh` | Cluster (Linux) | One-time setup: installs OCI CLI, downloads dataset + weights from Oracle, sets up Python venv |
| `train_h200.sh` | Cluster (SLURM) | Sbatch job: trains on H200 GPU(s), verbose epoch reports, uploads models to Oracle when done |
| `train_efficientnet_b0.py` | Cluster (Python) | Single-GPU training script |
| `train_efficientnet_b0_ddp.py` | Cluster (Python) | Multi-GPU DDP training script (used when GPUs > 1) |
| `epoch_report.py` | Cluster (Python) | Verbose every-5-epoch diagnostic report printed to SLURM log |
| `requirements.txt` | Cluster | pip dependencies — PyTorch 2.3+ with CUDA 12.4 (H200 compatible) |
| `dinov2/` | Cluster (Python) | DinoV2 backbone package — must be in `~/bloomi/` for imports to work |
| `New Data/train.txt` | Cluster (Python) | Pre-split training image paths (archive5, ~80%) |
| `New Data/val.txt` | Cluster (Python) | Pre-split validation image paths (archive5, ~20%) |

---

## Workflow (Windows)

### Step 1 — First-time setup
```cmd
.\setup.bat
```

This is the **only command you need to run on a fresh machine or cluster**. It:

1. Caches your SSH key passphrase — **enter it once**, all steps reuse it automatically
2. SCPs `setup.sh`, `train_h200.sh`, `epoch_report.py` → `~/`
3. SCPs `train_efficientnet_b0.py`, `train_efficientnet_b0_ddp.py`, `requirements.txt`, `epoch_report.py` → `~/bloomi/`
4. SCPs `dinov2/` → `~/bloomi/` (skips if already present)
5. SCPs `train.txt` + `val.txt` → `~/bloomi/New Data/` (skips if already present)
6. SCPs OCI private key → `~/.oci/oci_api_key.pem` (skips if already present)
7. Runs `setup.sh` on cluster — installs OCI CLI, downloads dataset + weights, creates Python venv

Setup is smart — skips anything already on the cluster:
- Skips `dinov2/` SCP if already present
- Skips `train.txt` / `val.txt` SCP if already present
- Skips OCI key SCP if already present
- Skips dataset download if >1000 images already in `~/bloomi/New Data/extracted/`
- Skips weights download if `DinoBloom-G.pth` already exists and is >100 MB
- Skips venv creation if `~/dinov2_venv` already exists

### Step 2 — Submit training job
```cmd
.\train_h200.bat          REM 1x H200 (default — normal QOS limit)
.\train_h200.bat 4        REM 4x H200 (requires unc_h200 QOS from sysadmin)
.\train_h200.bat 2        REM 2x H200
```

This caches your SSH passphrase, SCPs the latest training scripts to the cluster, and submits the sbatch job.

### Step 3 — Monitor
```cmd
ssh rpatel1@login-01.ncshare.org "squeue -u rpatel1"
ssh rpatel1@login-01.ncshare.org "tail -f ~/bloomi/logs/dino_<JOBID>.out"
```

---

## SLURM Job Config

| Setting | Value |
|---------|-------|
| GPUs | 1x H200 default (4x with `unc_h200` QOS) |
| GPUs VRAM | 143 GB per H200 |
| CPUs | 56 per task |
| RAM | 900 GB |
| Time limit | 48 hours (cluster max) |
| Partition | `gpu` |
| Batch size | 64/GPU |
| Workers | 8 per GPU (scales automatically) |

> **To get 4 GPUs:** Email the sysadmin to add `unc_h200` QOS to account `rpatel1` (ncssm). Default QOS only allows 1 GPU.

---

## Training Data Split

| Source | Split | How |
|--------|-------|-----|
| `archive5` (train.txt) | Training | Pre-defined split — 80% |
| `archive5` (val.txt) | Validation | Pre-defined split — 20% |
| `archive6`, `archive7`, `archive8` | Training + Validation | Random 80/20 split |

Archives 6/7/8 are expected in `~/bloomi/New Data/extracted/` on the cluster (downloaded by `setup.sh`).

---

## Epoch Summary Reports

Every **5 epochs**, a full diagnostic block is printed to the SLURM log (`logs/dino_<jobid>.out`):

- Timing: epoch time, avg/epoch, total elapsed, ETA, progress bar
- Loss & accuracy: train/val loss (8 decimal places), train/val acc, best so far
- Loss trend table: all recorded epochs side by side
- Optimizer state: LR, weight decay, betas, eps per param group
- Weight statistics: per layer — mean, std, L2 norm, min, max, % near-zero
- Gradient statistics: per layer — mean, std, norm, max, % NaN
- Global gradient norm + vanishing/exploding gradient warnings
- Model summary: total/trainable/frozen param counts, NaN/Inf/dead neuron checks
- GPU memory: allocated, reserved, peak, total VRAM per GPU

---

## Oracle ↔ H200 Flow

```
Oracle bucket: bloomi-training-data
    │
    │  setup.sh pulls DOWN:
    ├─ extracted/  (dataset, ~10 GB)
    └─ trained-models/dinobloom/dinobloom_g_finetuned.pth  (pretrained weights, ~4.4 GB)
    │
    │  [H200 trains for 75 epochs]
    │
    │  train_h200.sh pushes UP every epoch:
    ├─ trained-models/unc-h200/job<ID>_<date>/backups/checkpoint_epoch_NNN.pth
    │
    │  train_h200.sh pushes UP on completion:
    ├─ trained-models/unc-h200/job<ID>_<date>/best.pth   (best val accuracy)
    ├─ trained-models/unc-h200/job<ID>_<date>/last.pth   (final epoch)
    └─ trained-models/unc-h200/job<ID>_<date>/run_info.txt
```

No models accumulate on cluster disk — all checkpoints are uploaded to Oracle and deleted locally.

Download trained models via **Oracle Cloud Console**:

[cloud.oracle.com](https://cloud.oracle.com) → Storage → Object Storage → `bloomi-training-data` → `trained-models/` → `unc-h200/` → your job folder → download `best.pth`

---

## Requirements

- **Windows 10/11** — SSH and SCP are built in
- **OpenSSH Authentication Agent** — enabled automatically by `setup.bat` / `train_h200.bat` (passphrase cached per session)
- **OCI CLI** — installed automatically by `setup.sh` on the cluster
- **Python env** — `setup.sh` creates `~/dinov2_venv` using Python 3.12 venv + pip (no conda)

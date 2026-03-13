# clusterssh

Scripts for running DinoBloom-G leukemia classifier training on the ncshare H200 cluster.
**Windows only** for local/GitHub use (`.bat` files). Cluster runs Linux (`.sh` files).

Cluster: `rpatel1@login-01.ncshare.org`
Storage: Oracle Object Storage (`bloomi-training-data`, region `us-ashburn-1`)

---

## Files

| File | Where it runs | What it does |
|------|--------------|--------------|
| `setup.bat` | Windows (local) | SCPs all scripts + training code to cluster, runs setup remotely |
| `train_h200.bat` | Windows (local) | SCPs latest training code to cluster, submits `sbatch` job |
| `monitor.bat` | Windows (local) | Live dashboard — GPU stats, SLURM queue, training metrics |
| `setup.sh` | Cluster (Linux) | One-time setup: installs OCI CLI, downloads dataset + weights from Oracle, sets up conda env |
| `train_h200.sh` | Cluster (SLURM) | Sbatch job: trains on 4x H200, prints verbose epoch reports, uploads models to Oracle when done |
| `train_efficientnet_b0.py` | Cluster (Python) | Single-GPU training script |
| `train_efficientnet_b0_ddp.py` | Cluster (Python) | Multi-GPU DDP training script (used by default with 4x H200) |
| `conda.yaml` | Cluster | Conda environment definition (`dinov2` env) |
| `epoch_report.py` | Cluster (Python) | Imported by training scripts — prints exhaustive every-5-epoch summary to SLURM console |
| `_monitor_remote.sh` | Cluster (piped via SSH) | Helper script piped by `monitor.bat` — keep in same folder |

---

## Workflow (Windows)

### Step 1 — First-time setup
```cmd
.\setup.bat
```
This SCPs all scripts and training code to the cluster, then runs setup remotely.

Setup does:
- SCPs `setup.sh`, `train_h200.sh`, `epoch_report.py` → `~/`
- SCPs `train_efficientnet_b0.py`, `train_efficientnet_b0_ddp.py`, `conda.yaml`, `epoch_report.py` → `~/bloomi/`
- SCPs `.pem` key → `~/.oci/oci_api_key.pem`
- Detects or installs OCI CLI (verbose output, pip fallback if needed)
- Downloads dataset (~10 GB) from Oracle bucket
- Downloads DinoBloom-G pretrained weights (~4.4 GB) from Oracle
- Creates `dinov2` conda environment

### Step 2 — Submit training job
```cmd
.\train_h200.bat          REM 4x H200 (default)
.\train_h200.bat 2        REM 2x H200
.\train_h200.bat 1        REM 1x H200
```
This SCPs the latest `train_h200.sh` and `epoch_report.py` to the cluster and runs `sbatch`.

### Step 3 — Monitor
```cmd
.\monitor.bat             REM live dashboard (GPU, queue, metrics, log tail)
.\monitor.bat connect     REM plain SSH shell into cluster
```

---

## SLURM Job Config (4x H200)

| Setting | Value |
|---------|-------|
| GPUs | 4x H200 (96 GB VRAM each = 384 GB total) |
| CPUs | 56 per task |
| RAM | 900 GB |
| Time limit | 96 hours |
| Partition | `gpu_p` |
| Constraint | `h200` |
| Batch size | 64/GPU → 256 effective |
| Workers | 224 dataloader workers |

---

## Epoch Summary Reports

Every **5 epochs**, the training script prints a full diagnostic block to the SLURM console log (`logs/dino_<jobid>.out`). Each report includes:

- Timing: epoch time, avg time/epoch, total elapsed, ETA, progress bar
- Loss & accuracy: train/val loss (8 decimal places), train/val acc, best so far
- Loss trend table: all recorded epochs side by side
- Optimizer state: type, learning rate, weight decay, betas, eps per param group
- Weight statistics: per layer — mean, std, L2 norm, min, max, % near-zero
- Gradient statistics: per layer — mean, std, norm, max, % NaN
- Global gradient norm + vanishing/exploding gradient warnings
- Model summary: total/trainable/frozen param counts, NaN/Inf/dead neuron checks
- GPU memory: allocated, reserved, peak, total VRAM per GPU

### Integration in training scripts

Add to your training script (`train_efficientnet_b0.py` / `_ddp.py`):

```python
from epoch_report import EpochReporter
reporter = EpochReporter(report_every=5)

for epoch in range(1, args.epochs + 1):
    reporter.epoch_start()
    # ... train loop ...
    # ... validation loop ...
    reporter.report(
        epoch=epoch,
        total_epochs=args.epochs,
        model=model,
        optimizer=optimizer,
        train_loss=train_loss,
        val_loss=val_loss,
        train_acc=train_acc,   # optional, float 0-1
        val_acc=val_acc,       # optional, float 0-1
        extra={"lr": scheduler.get_last_lr()[0]}  # any extra key-value pairs
    )
```

---

## Oracle ↔ H200 Flow

```
Oracle bucket: bloomi-training-data
    │
    │  setup.sh pulls DOWN:
    ├─ extracted/  (dataset, ~10 GB)
    └─ DinoBloom-G.pth  (pretrained weights, ~4.4 GB)
    │
    │  [H200 trains for 75 epochs on 4x H200 GPUs]
    │
    │  train_h200.sh pushes UP:
    ├─ trained-models/unc-h200/job<ID>_<date>/best.pth
    └─ trained-models/unc-h200/job<ID>_<date>/last.pth
```

Download trained models via **Oracle Cloud Console** (easiest):

[cloud.oracle.com](https://cloud.oracle.com) → Storage → Object Storage → `bloomi-training-data` → `trained-models/` → `unc-h200/` → your job folder → download `best.pth`

---

## Requirements

- **Windows 10/11** — SSH and SCP are built in (open CMD and run `ssh` to verify)
- **PowerShell** — always prefix scripts with `.\` (e.g. `.\setup.bat`)
- **OCI CLI** — installed automatically by `setup.sh` if missing (verbose output shown)
- **conda** — set up automatically by `setup.sh`

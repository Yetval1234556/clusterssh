# Cluster SSH — DinoBloom-G Training Setup

Scripts for connecting to a university HPC cluster and launching DinoBloom-G leukemia classifier training using SLURM.

## Files

| File | Description |
|------|-------------|
| `connect.bat` | Windows batch file to SSH into the cluster |
| `setup.sh` | One-time setup script — pulls data, clones repo, submits job |
| `train_cluster_1gpu.sh` | SLURM job script — single GPU, standard training |
| `train_cluster_8gpu.sh` | SLURM job script — 8 GPUs with torchrun + DDP (faster) |

The multi-GPU script requires `train_efficientnet_b0_ddp.py` from the [DinoModelsEXTRA](https://github.com/Yetval1234556/DinoModelsEXTRA) repo.

## Quick Start

### 1. Connect to the cluster
Edit `connect.bat` and replace `YOUR_CLUSTER.unc.edu` with your cluster address, then double-click.

### 2. Run setup on the cluster
```bash
bash setup.sh
```
This will:
- Configure Oracle Object Storage credentials
- Clone the DinoModelsEXTRA repo to scratch storage
- Download the 10GB dataset from Oracle Object Storage
- Set up the `dinov2` conda environment
- Submit the training job

### 3. Choose your training mode

**Single GPU** — simpler, uses `train_efficientnet_b0.py`:
```bash
sbatch train_cluster_1gpu.sh
```

**8 GPUs (recommended)** — uses `train_efficientnet_b0_ddp.py` with torchrun + DistributedDataParallel:
```bash
sbatch train_cluster_8gpu.sh
```
Effective batch size with 8 GPUs = 8 (per GPU) × 8 (GPUs) = **64** — significantly faster per epoch.

### 4. Monitor your job
```bash
squeue -u $USER
tail -f logs/run_<job_id>.txt
```

## Cluster Specs (configured for)
| Setting | Value |
|---------|-------|
| Partition | `gpu_p` |
| QOS | `gpu_reservation` |
| Reservation | `test_supergpu05` |
| Max time | 96 hours |
| CPUs (8-GPU job) | 126 |
| RAM (8-GPU job) | 1800 GB |

## Requirements
- AWS CLI on the cluster (`aws --version`)
- Conda (`module load conda` or available in `$HOME/.bashrc`)
- CUDA 11.7+
- SLURM scheduler

## Retrieving the Trained Model

After training completes, the model is **automatically uploaded** to Oracle Object Storage:

```
bloomi-training-data/trained-models/dinobloom_g_leukemia_classifier_YYYYMMDD_HHMMSS.pth
```

- The `trained-models/` prefix acts as a folder inside the `bloomi-training-data` bucket
- Each run gets a unique timestamp so nothing is ever overwritten
- Browse it in the [Oracle Cloud Console](https://cloud.oracle.com) → Object Storage → `bloomi-training-data` → `trained-models/`

To download to your PC after training:
```bash
oci os object get \
  --namespace idcsxwupyymi \
  --bucket-name bloomi-training-data \
  --name "trained-models/<filename>.pth" \
  --file "C:\Users\19802\Downloads\bloomi extra\dinobloom_g_leukemia_classifier.pth"
```

## Notes
- DinoBloom-G pretrained weights (`DinoBloom-G.pth`, ~4.4GB) must be added to setup.sh — see the placeholder in step 4
- Dataset is pulled from Oracle Object Storage (`bloomi-training-data` bucket, namespace `idcsxwupyymi`)
- Training saves best model to `dinobloom_g_finetuned.pth` and per-epoch backups to `backup/`

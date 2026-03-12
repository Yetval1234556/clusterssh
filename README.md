# Cluster SSH — DinoBloom-G Training Setup

Scripts for connecting to a university HPC cluster and launching DinoBloom-G leukemia classifier training using SLURM.

## Files

| File | Description |
|------|-------------|
| `connect.bat` | Windows batch file to SSH into the cluster |
| `setup.sh` | One-time setup script — pulls data, clones repo, submits job |
| `train_cluster.sh` | SLURM job script for training on an A100 GPU |

## Quick Start

### 1. Connect to the cluster
Edit `connect.bat` and replace `YOUR_CLUSTER.unc.edu` with your cluster's address, then double-click to connect.

### 2. Run setup on the cluster
```bash
bash setup.sh
```
This will:
- Configure Oracle Object Storage credentials
- Clone the [DinoModelsEXTRA](https://github.com/Yetval1234556/DinoModelsEXTRA) repo
- Download the 10GB dataset from Oracle Object Storage
- Set up the `dinov2` conda environment
- Submit the SLURM training job

### 3. Monitor your job
```bash
squeue -u $USER
```

### 4. View training logs
```bash
tail -f logs/train_<job_id>.out
```

## Requirements
- AWS CLI installed on the cluster (`aws --version`)
- Conda available (`module load conda`)
- CUDA 11.7+ (`module load cuda`)
- SLURM scheduler

## Notes
- Update `--partition` in `train_cluster.sh` to match your cluster's GPU partition name
- DinoBloom-G pretrained weights (`DinoBloom-G.pth`) must be downloaded separately — see step 4 in `setup.sh`
- Dataset is pulled from Oracle Object Storage (`bloomi-training-data` bucket)
- Training runs 30 epochs with batch size 16 on a single A100 (~3-4 hours)

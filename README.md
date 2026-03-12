# clusterssh

Scripts for running DinoBloom-G leukemia classifier training on remote GPU servers.

Supports two environments:
- **UNC H200 cluster** — 8x H200 GPUs via SLURM
- **Oracle A100 VM** — 1-2x A100 GPUs, no job scheduler

---

## Files

| File | What it does |
|------|-------------|
| `monitor.sh` | Live GPU + training monitor from your Mac. Auto-reconnects if internet drops. |
| `setup.sh` | One-time cluster setup — pulls repo, dataset, weights, sets up conda |
| `train_unc_h200.sh` | SLURM job for UNC H200 cluster |
| `train_oracle_a100.sh` | Training script for Oracle A100 VM (runs inside tmux) |

---

## Quick Start

### UNC H200

```bash
# 1. SSH in and run setup once
bash setup.sh

# 2. Submit job
sbatch train_unc_h200.sh        # 8x H200
sbatch train_unc_h200.sh 1      # 1x H200

# 3. Monitor from your Mac
./monitor.sh unc
./monitor.sh unc connect        # plain SSH
```

### Oracle A100

```bash
# 1. SSH in, clone repo, run setup.sh

# 2. Start training (runs in tmux automatically)
bash train_oracle_a100.sh 1     # 1x A100
bash train_oracle_a100.sh 2     # 2x A100

# 3. Monitor from your Mac
./monitor.sh oracle
./monitor.sh oracle connect     # plain SSH

# If your Mac loses internet, reconnect and resume:
tmux attach -t dinobloom
```

---

## Trained Models

Best model is **automatically uploaded** to Oracle Object Storage after each accuracy improvement. Previous model is deleted.

```
bloomi-training-data/trained-models/dinobloom_g_epoch<N>_acc<X>.pth
```

Download when done:
```bash
oci os object get \
  --namespace idcsxwupyymi \
  --bucket-name bloomi-training-data \
  --name "trained-models/<filename>.pth" \
  --file ~/Downloads/dinobloom_finetuned.pth
```

Browse at [Oracle Cloud Console](https://cloud.oracle.com) → Object Storage → `bloomi-training-data` → `trained-models/`

---

## Requirements

- OCI CLI (`pip install oci-cli`) — credentials hardcoded in `setup.sh`
- Copy `~/.oci/oci_api_key.pem` to the cluster before running `setup.sh`
- `conda` with `dinov2` environment (created by `setup.sh`)
- `tmux` on Oracle VM (`sudo yum install -y tmux`)

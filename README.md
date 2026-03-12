# clusterssh

Scripts for running DinoBloom-G leukemia classifier training on remote GPU servers from your Mac.

Supports two environments:
- **UNC H200 cluster** — 8x H200 GPUs, managed by SLURM job scheduler
- **Oracle A100 VM** — 1-2x A100 GPUs, runs directly (no job scheduler)

---

## Files

| File | What it does |
|------|-------------|
| `monitor.sh` | Run from your Mac — shows live GPU stats and training progress. Auto-reconnects if your internet drops. |
| `setup.sh` | Run once on the cluster — downloads the dataset, model weights, and sets up the conda environment. |
| `train_unc_h200.sh` | Submits a training job to the UNC SLURM scheduler. |
| `train_oracle_a100.sh` | Starts training on the Oracle VM inside a tmux session (keeps running if your Mac disconnects). |

---

## UNC H200

```bash
# On the cluster — first time only
bash setup.sh

# Submit training job
sbatch train_unc_h200.sh        # 8x H200 (recommended, effective batch size 512)
sbatch train_unc_h200.sh 1      # 1x H200

# From your Mac — live dashboard (GPU %, memory, training metrics, recent logs)
./monitor.sh unc

# From your Mac — plain SSH with auto-reconnect
./monitor.sh unc connect
```

---

## Oracle A100

```bash
# On the VM — first time only
bash setup.sh

# Start training — automatically runs inside tmux so it keeps going if you disconnect
bash train_oracle_a100.sh 1     # 1x A100 80GB
bash train_oracle_a100.sh 2     # 2x A100

# From your Mac — live dashboard
./monitor.sh oracle

# From your Mac — plain SSH with auto-reconnect
./monitor.sh oracle connect

# If your Mac lost internet and you need to get back to the training session
ssh opc@<instance-ip>
tmux attach -t dinobloom
```

---

## Getting the Trained Model

The best model is **automatically saved to Oracle Object Storage** every time test accuracy improves. The previous best is deleted so only one copy is kept.

Saved to:
```
bloomi-training-data/trained-models/dinobloom_g_epoch<N>_acc<X>.pth
```

Download to your Mac when training is done:
```bash
oci os object get \
  --namespace idcsxwupyymi \
  --bucket-name bloomi-training-data \
  --name "trained-models/<filename>.pth" \
  --file ~/Downloads/dinobloom_finetuned.pth
```

Or browse the files at [Oracle Cloud Console](https://cloud.oracle.com) → Object Storage → `bloomi-training-data` → `trained-models/`

---

## Requirements

- **OCI CLI** on your Mac and on the cluster (`pip install oci-cli`) — credentials are hardcoded in `setup.sh`, no manual key setup needed
- **OCI API private key** (`~/.oci/oci_api_key.pem`) must be copied to the cluster before running `setup.sh`:
  ```bash
  scp ~/.oci/oci_api_key.pem YOUR_CLUSTER.unc.edu:~/.oci/oci_api_key.pem
  ```
- **conda** — `setup.sh` creates the `dinov2` environment automatically
- **tmux** on Oracle VM — usually pre-installed, or `sudo yum install -y tmux`

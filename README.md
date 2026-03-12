# clusterssh

Scripts for running DinoBloom-G leukemia classifier training on remote GPU servers from your Mac.

Supports two environments:
- **UNC H200 cluster** — 8x H200 GPUs, managed by SLURM job scheduler
- **Oracle A100 VM** — 1-2x A100 GPUs, runs directly (no job scheduler)

---

## Files

| File | What it does |
|------|-------------|
| `monitor.sh` | Run from your Mac — live GPU stats and training progress. Auto-reconnects if internet drops. (`unc connect` / `oracle connect` for plain SSH.) |
| `setup.sh` | Run once on the cluster — downloads dataset, model weights, sets up conda. |
| `train_unc_h200.sh` | Submits training job to UNC SLURM scheduler. |
| `train_oracle_a100.sh` | Starts training on Oracle VM inside tmux (keeps running if Mac disconnects). |
| `train_efficientnet_b0.py` | Single-GPU training script — saves `best.pth` and `last.pth` to Oracle bucket each epoch. |
| `train_efficientnet_b0_ddp.py` | Multi-GPU DDP training script — same save logic, launched via torchrun. |

---

## UNC H200

**Step 1 — SSH into the cluster and run setup (first time only):**
```bash
ssh YOUR_USER@YOUR_CLUSTER.unc.edu
bash setup.sh
```

**Step 2 — Submit the training job (still on the cluster):**
```bash
sbatch train_unc_h200.sh        # 8x H200 (recommended, effective batch size 512)
sbatch train_unc_h200.sh 1      # 1x H200
```

**Step 3 — Monitor from your Mac:**
```bash
./monitor.sh unc                # live dashboard — GPU %, memory, training metrics, recent logs
```

---

## Oracle A100

**Step 1 — SSH into the VM and run setup (first time only):**
```bash
ssh opc@<instance-ip>
bash setup.sh
```

**Step 2 — Start training (still on the VM):**
```bash
bash train_oracle_a100.sh 1     # 1x A100 80GB
bash train_oracle_a100.sh 2     # 2x A100
```
Training runs inside tmux automatically — safe to close the terminal.

**Step 3 — Monitor from your Mac:**
```bash
./monitor.sh oracle             # live dashboard
```

**If your Mac lost internet and training is still running:**
```bash
ssh opc@<instance-ip>
tmux attach -t dinobloom
```

---

## Trained Models

Every epoch, two files are automatically saved to Oracle Object Storage:
- `best.pth` — best test accuracy so far (use this to deploy)
- `last.pth` — full checkpoint every epoch including optimizer state (use this to resume)

Bucket layout:
```
bloomi-training-data/
  trained-models/
    unc-h200/
      job<SLURM_ID>_<date>/
        best.pth
        last.pth
    oracle-a100/
      <date>/
        best.pth
        last.pth
```

Download to your Mac when done:
```bash
# UNC run
oci os object get \
  --namespace idcsxwupyymi \
  --bucket-name bloomi-training-data \
  --name "trained-models/unc-h200/job<ID>_<date>/best.pth" \
  --file ~/Downloads/dinobloom_best.pth

# Oracle run
oci os object get \
  --namespace idcsxwupyymi \
  --bucket-name bloomi-training-data \
  --name "trained-models/oracle-a100/<date>/best.pth" \
  --file ~/Downloads/dinobloom_best.pth
```

Browse all runs at [Oracle Cloud Console](https://cloud.oracle.com) → Object Storage → `bloomi-training-data` → `trained-models/`

---

## Requirements

- **OCI CLI** on your Mac and on the cluster (`pip install oci-cli`) — credentials are hardcoded, no manual key setup needed
- **OCI API private key** (`~/.oci/oci_api_key.pem`) must be copied to the cluster before running `setup.sh`:
  ```bash
  scp ~/.oci/oci_api_key.pem YOUR_CLUSTER.unc.edu:~/.oci/oci_api_key.pem
  ```
- **conda** — `setup.sh` creates the `dinov2` environment automatically
- **tmux** on Oracle VM — usually pre-installed, or `sudo yum install -y tmux`

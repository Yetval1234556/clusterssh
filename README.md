# clusterssh

Scripts for running DinoBloom-G leukemia classifier training on remote GPU servers from your Mac.

Supports two environments:
- **UNC H200 cluster** — 8x H200 GPUs, managed by SLURM job scheduler
- **Oracle A100 VM** — 1-2x A100 GPUs, runs directly (no job scheduler)

---

## Files

| File | What it does |
|------|-------------|
| `monitor.sh` | Run from your Mac — shows live GPU stats and training progress. Auto-reconnects if your internet drops. (`./monitor.sh unc connect` / `./monitor.sh oracle connect` opens a plain SSH session instead.) |
| `setup.sh` | Run once on the cluster — downloads the dataset, model weights, and sets up the conda environment. |
| `train_unc_h200.sh` | Submits a training job to the UNC SLURM scheduler. |
| `train_oracle_a100.sh` | Starts training on the Oracle VM inside a tmux session (keeps running if your Mac disconnects). |

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

## Getting the Trained Model

Every epoch, two files are saved to Oracle Object Storage:
- `best.pth` — best test accuracy so far (use this to deploy)
- `last.pth` — latest checkpoint (use this to resume if training is interrupted)

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

Download to your Mac:
```bash
# Best model (UNC run)
oci os object get \
  --namespace idcsxwupyymi \
  --bucket-name bloomi-training-data \
  --name "trained-models/unc-h200/job<ID>_<date>/best.pth" \
  --file ~/Downloads/dinobloom_best.pth

# Best model (Oracle run)
oci os object get \
  --namespace idcsxwupyymi \
  --bucket-name bloomi-training-data \
  --name "trained-models/oracle-a100/<date>/best.pth" \
  --file ~/Downloads/dinobloom_best.pth
```

Browse all runs at [Oracle Cloud Console](https://cloud.oracle.com) → Object Storage → `bloomi-training-data` → `trained-models/`

---

## Requirements

- **OCI CLI** on your Mac and on the cluster (`pip install oci-cli`) — credentials are hardcoded in `setup.sh`, no manual key setup needed
- **OCI API private key** (`~/.oci/oci_api_key.pem`) must be copied to the cluster before running `setup.sh`:
  ```bash
  scp ~/.oci/oci_api_key.pem YOUR_CLUSTER.unc.edu:~/.oci/oci_api_key.pem
  ```
- **conda** — `setup.sh` creates the `dinov2` environment automatically
- **tmux** on Oracle VM — usually pre-installed, or `sudo yum install -y tmux`

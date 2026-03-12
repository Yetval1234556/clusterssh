# clusterssh

Scripts for running DinoBloom-G leukemia classifier training on remote GPU servers.
Works on **Mac** (`.sh`) and **Windows** (`.bat`).

Cluster: `rpatel1@ncshare.login.org`

Supports two environments:
- **UNC H200 cluster** — 8x H200 GPUs, managed by SLURM job scheduler
- **Oracle A100 VM** — 1-2x A100 GPUs, runs directly (no job scheduler)

---

## Files

| Mac | Windows | What it does |
|-----|---------|--------------|
| `monitor.sh` | `monitor.bat` | Run locally — live GPU stats and training progress. Auto-reconnects if internet drops. |
| `setup.sh` | `setup.bat` | Run once — SCPs setup script to cluster and runs it (downloads dataset, weights, sets up conda). |
| `train_unc_h200.sh` | `train_unc_h200.bat` | Submits training job to UNC SLURM scheduler. |
| `train_oracle_a100.sh` | `train_oracle_a100.bat` | Starts training on Oracle VM inside tmux (keeps running if you disconnect). |
| `_monitor_remote.sh` | *(companion to monitor.bat)* | Remote bash commands piped over SSH — keep in same folder as `monitor.bat`. |

---

## Step 0 — SCP scripts to the server

Before running anything on the cluster, copy the scripts over from your local machine.

**Mac / Linux:**
```bash
scp setup.sh train_unc_h200.sh train_oracle_a100.sh rpatel1@ncshare.login.org:~/

# Also copy your OCI API key (needed for dataset download):
scp ~/.oci/oci_api_key.pem rpatel1@ncshare.login.org:~/.oci/oci_api_key.pem
```

**Windows (Command Prompt):**
```cmd
scp setup.sh train_unc_h200.sh train_oracle_a100.sh rpatel1@ncshare.login.org:/home/rpatel1/

rem Also copy your OCI API key:
scp %USERPROFILE%\.oci\oci_api_key.pem rpatel1@ncshare.login.org:/home/rpatel1/.oci/oci_api_key.pem
```

> For Oracle, update `ORACLE_HOST` in `monitor.sh` / `monitor.bat` with your VM's public IP.

---

## UNC H200

### Mac

**Step 1 — Run setup on the cluster (first time only):**
```bash
ssh rpatel1@ncshare.login.org
bash ~/setup.sh
```

**Step 2 — Submit the training job (still on the cluster):**
```bash
sbatch ~/train_unc_h200.sh        # 8x H200 (recommended)
sbatch ~/train_unc_h200.sh 1      # 1x H200
```

**Step 3 — Monitor from your Mac:**
```bash
./monitor.sh unc                  # live dashboard
./monitor.sh unc connect          # plain SSH into cluster
```

### Windows

**Steps 1 & 2 are handled automatically by the .bat files — run from Command Prompt:**
```cmd
setup.bat                         # SCPs setup.sh and runs it on the cluster
train_unc_h200.bat                # SCPs train script and submits with sbatch
train_unc_h200.bat 1              # 1x H200
```

**Step 3 — Monitor:**
```cmd
monitor.bat unc                   # live dashboard
monitor.bat unc connect           # plain SSH into cluster
```

---

## Oracle A100

### Mac

**Step 1 — Run setup on the VM (first time only):**
```bash
ssh opc@<instance-ip>
bash ~/setup.sh
```

**Step 2 — Start training (still on the VM):**
```bash
bash ~/train_oracle_a100.sh 1     # 1x A100 80GB
bash ~/train_oracle_a100.sh 2     # 2x A100
```
Training runs inside tmux automatically — safe to close the terminal.

**Step 3 — Monitor from your Mac:**
```bash
./monitor.sh oracle               # live dashboard
./monitor.sh oracle connect       # plain SSH into Oracle VM
```

**If your Mac lost internet and training is still running:**
```bash
ssh opc@<instance-ip>
tmux attach -t dinobloom
```

### Windows

```cmd
setup.bat                         # (edit ORACLE_HOST in setup.bat first)
train_oracle_a100.bat             # 1x A100
train_oracle_a100.bat 2           # 2x A100
monitor.bat oracle                # live dashboard
monitor.bat oracle connect        # plain SSH
```

**If you got disconnected and training is still running:**
```cmd
ssh opc@<instance-ip>
```
Then on the server: `tmux attach -t dinobloom`

---

## Trained Models

Every epoch, two files are automatically saved to Oracle Object Storage:
- `best.pth` — best test accuracy so far (use this to deploy)
- `last.pth` — full checkpoint including optimizer state (use this to resume)

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

- **SSH / SCP** — built into Mac/Linux. On Windows, built into Windows 10/11 (open CMD and type `ssh` to verify).
- **OCI CLI** — install on your local machine and on the cluster: `pip install oci-cli`
- **OCI API private key** — `~/.oci/oci_api_key.pem` must exist locally and be SCP'd to the cluster (see Step 0 above)
- **conda** — `setup.sh` creates the `dinov2` environment automatically on the cluster
- **tmux** — on Oracle VM, usually pre-installed. If not: `sudo yum install -y tmux`

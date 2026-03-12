# clusterssh

Scripts for running DinoBloom-G leukemia classifier training on the ncshare H200 cluster.
Works on **Mac** (`.sh`) and **Windows** (`.bat`).

Cluster: `rpatel1@login-01.ncshare.org`
Storage: Oracle Object Storage (`bloomi-training-data`)

---

## Files

| Mac | Windows | What it does |
|-----|---------|--------------|
| `monitor.sh` | `monitor.bat` | Run locally — live GPU stats, SLURM queue, training metrics. Auto-reconnects. |
| `setup.sh` | `setup.bat` | Run once — configures OCI, downloads dataset + weights from Oracle bucket, sets up conda. |
| `train_h200.sh` | `train_h200.bat` | Submits training job to SLURM on the H200 cluster. Uploads models to Oracle bucket when done. |
| `_monitor_remote.sh` | *(companion to monitor.bat)* | Remote commands piped over SSH — keep in same folder as `monitor.bat`. |

---

## Step 0 — SCP scripts to the cluster

**Mac / Linux:**
```bash
scp setup.sh train_h200.sh rpatel1@login-01.ncshare.org:~/
```

**Windows (PowerShell — always use `.\`):**
```cmd
.\setup.bat    # automatically SCPs and runs setup for you
```

---

## Usage

### Mac

**Step 1 — Run setup (first time only):**
```bash
ssh rpatel1@login-01.ncshare.org
bash ~/setup.sh
```

**Step 2 — Submit training job:**
```bash
sbatch ~/train_h200.sh        # 2x H200 (default)
sbatch ~/train_h200.sh 1      # 1x H200
```

**Step 3 — Monitor from your Mac:**
```bash
./monitor.sh                  # live dashboard
./monitor.sh connect          # plain SSH into cluster
```

---

### Windows (PowerShell)

**Step 1 — Run setup:**
```cmd
.\setup.bat
```

**Step 2 — Submit training job:**
```cmd
.\train_h200.bat              # 2x H200 (default)
.\train_h200.bat 1            # 1x H200
```

**Step 3 — Monitor:**
```cmd
.\monitor.bat                 # live dashboard
.\monitor.bat connect         # plain SSH into cluster
```

---

## Trained Models

Every epoch, models are saved to Oracle Object Storage:
- `best.pth` — best test accuracy so far
- `last.pth` — full checkpoint with optimizer state (use to resume)

Bucket layout:
```
bloomi-training-data/
  trained-models/
    h200/
      job<SLURM_ID>_<date>/
        best.pth
        last.pth
```

Download when done:
```bash
oci os object get \
  --namespace idcsxwupyymi \
  --bucket-name bloomi-training-data \
  --name "trained-models/h200/job<ID>_<date>/best.pth" \
  --file ~/Downloads/dinobloom_best.pth
```

Browse all runs at [Oracle Cloud Console](https://cloud.oracle.com) → Object Storage → `bloomi-training-data` → `trained-models/`

---

## Requirements

- **SSH / SCP** — built into Mac/Linux. On Windows 10/11, open CMD and type `ssh` to verify.
- **PowerShell** — always prefix with `.\` (e.g. `.\setup.bat`, `.\monitor.bat`)
- **OCI CLI** — installed automatically by `setup.sh`. Credentials are hardcoded — no key file needed.
- **conda** — set up automatically by `setup.sh`.

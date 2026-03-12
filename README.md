# clusterssh

Scripts for running DinoBloom-G leukemia classifier training on an Oracle A100 GPU server.
Works on **Mac** (`.sh`) and **Windows** (`.bat`).

Cluster: `rpatel1@login-01.ncshare.org`

---

## Files

| Mac | Windows | What it does |
|-----|---------|--------------|
| `monitor.sh` | `monitor.bat` | Run locally — live GPU stats and training progress. Auto-reconnects if internet drops. |
| `setup.sh` | `setup.bat` | Run once — sets up OCI, downloads dataset + weights, installs conda env on Oracle VM. |
| `train_oracle_a100.sh` | `train_oracle_a100.bat` | Starts training on Oracle VM inside tmux (keeps running if you disconnect). |
| `_monitor_remote.sh` | *(companion to monitor.bat)* | Remote bash commands piped over SSH — keep in same folder as `monitor.bat`. |

---

## Step 0 — SCP scripts to Oracle VM

Copy the scripts to your Oracle VM before running them.

**Mac / Linux:**
```bash
scp setup.sh train_oracle_a100.sh opc@YOUR_ORACLE_INSTANCE_IP:~/
```

**Windows (Command Prompt or PowerShell):**
```cmd
scp setup.sh train_oracle_a100.sh opc@YOUR_ORACLE_INSTANCE_IP:/home/opc/
```

> Replace `YOUR_ORACLE_INSTANCE_IP` with your Oracle VM's public IP. Also update `ORACLE_HOST` in `monitor.sh` / `monitor.bat`.

---

## Oracle A100

### Mac

**Step 1 — Run setup on the VM (first time only):**
```bash
ssh opc@YOUR_ORACLE_INSTANCE_IP
bash ~/setup.sh
```

**Step 2 — Start training:**
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

**If you got disconnected and training is still running:**
```bash
ssh opc@YOUR_ORACLE_INSTANCE_IP
tmux attach -t dinobloom
```

---

### Windows

**Step 1 — Update your Oracle IP** in `setup.bat` and `monitor.bat` (`ORACLE_HOST=...`).

**Step 2 — Run setup (SCPs and runs setup.sh on Oracle automatically):**
```cmd
.\setup.bat
```

**Step 3 — Start training:**
```cmd
.\train_oracle_a100.bat           # 1x A100
.\train_oracle_a100.bat 2         # 2x A100
```

**Step 4 — Monitor:**
```cmd
.\monitor.bat oracle              # live dashboard
.\monitor.bat oracle connect      # plain SSH
```

**If you got disconnected and training is still running:**
```cmd
ssh opc@YOUR_ORACLE_INSTANCE_IP
```
Then on the VM: `tmux attach -t dinobloom`

---

## Trained Models

Every epoch, two files are automatically saved to Oracle Object Storage:
- `best.pth` — best test accuracy so far (use this to deploy)
- `last.pth` — full checkpoint including optimizer state (use this to resume)

Bucket layout:
```
bloomi-training-data/
  trained-models/
    oracle-a100/
      <date>/
        best.pth
        last.pth
```

Download when done:
```bash
oci os object get \
  --namespace idcsxwupyymi \
  --bucket-name bloomi-training-data \
  --name "trained-models/oracle-a100/<date>/best.pth" \
  --file ~/Downloads/dinobloom_best.pth
```

Browse all runs at [Oracle Cloud Console](https://cloud.oracle.com) → Object Storage → `bloomi-training-data` → `trained-models/`

---

## Requirements

- **SSH / SCP** — built into Mac/Linux. On Windows 10/11, built in — open CMD/PowerShell and type `ssh` to verify.
- **Always use `.\` in PowerShell** when running `.bat` files (e.g. `.\setup.bat`, `.\monitor.bat oracle`).
- **OCI CLI** — installed automatically by `setup.sh` on the Oracle VM. Credentials are hardcoded — no key file needed.
- **conda** — installed automatically by `setup.sh` if not already present.
- **tmux** — on Oracle VM, usually pre-installed. If not: `sudo yum install -y tmux`

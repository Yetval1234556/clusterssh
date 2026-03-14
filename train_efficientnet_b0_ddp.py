#!/usr/bin/env python3
"""
Multi-GPU leukemia classifier built on top of the original DinoBloom-G pretrained model.

Input  (base): DinoBloom-G.pth  — original pretrained model (from Google Drive).
               ViT-Giant trained on 13M+ hematology images. We do NOT modify this file.
Output (ours): bloom_leukemia.pth — our model with the leukemia head baked in.

Launch with:
    torchrun --nproc_per_node=4 train_efficientnet_b0_ddp.py [args]
"""

import os
import sys
import argparse
import json
import csv
import time
import datetime
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from PIL import Image
from pathlib import Path
import random
import warnings

REPO_ROOT     = Path(__file__).parent.resolve()
OCI_NAMESPACE = "idcsxwupyymi"
OCI_BUCKET    = "bloomi-training-data"

# Organize bucket by environment and SLURM job ID
_run_date    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
_slurm_job   = os.environ.get("SLURM_JOB_ID", "")
if _slurm_job:
    OCI_RUN_PREFIX = f"trained-models/unc-h200/job{_slurm_job}_{_run_date}"
else:
    OCI_RUN_PREFIX = f"trained-models/oracle-a100/{_run_date}"

def _oci_bin():
    import shutil, os
    oci = shutil.which("oci")
    if oci:
        return oci
    for p in [
        os.path.expanduser("~/.local/bin/oci"),
        os.path.expanduser("~/bin/oci"),
        "/usr/local/bin/oci",
    ]:
        if os.path.isfile(p):
            return p
    return "oci"

def oracle_upload(local_path: str, object_name: str):
    env = os.environ.copy()
    env["PATH"] = os.path.expanduser("~/.local/bin") + ":" + env.get("PATH", "")
    result = subprocess.run([
        _oci_bin(), "os", "object", "put",
        "--namespace", OCI_NAMESPACE,
        "--bucket-name", OCI_BUCKET,
        "--name", object_name,
        "--file", local_path,
        "--force"
    ], env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"oci put failed: {result.stderr.strip()}")
    print(f"  [oracle] uploaded → {OCI_BUCKET}/{object_name}")

def oracle_delete(object_name: str):
    subprocess.run([
        "oci", "os", "object", "delete",
        "--namespace", OCI_NAMESPACE,
        "--bucket-name", OCI_BUCKET,
        "--object-name", object_name,
        "--force"
    ], check=True)
    print(f"  [oracle] deleted  → {OCI_BUCKET}/{object_name}")
VALID_EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

EXCLUDE_FOLDERS = {
    "C-NMC_test_final_phase_data",
    "C-NMC_test_prelim_phase_data",
    "testing_data",
    "training_data",
    "extracted",
}

warnings.filterwarnings("ignore", message="xFormers is not available")


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def is_main():
    return dist.get_rank() == 0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def class_from_path(path: Path) -> str:
    """
    Extract class label using the C-NMC filename convention:
        {patient_id}_{slide_id}_{image_id}_{CLASS}.ext
        e.g.  4_7_400_ALL.bmp  →  ALL
              UID_1_1_hem.bmp  →  hem

    Falls back to parent folder name when the last underscore segment
    is purely numeric or the filename has no underscores.
    """
    parts     = path.stem.split("_")
    candidate = parts[-1]
    if candidate and any(c.isalpha() for c in candidate):
        return candidate
    return path.parent.name


def load_txt_samples(txt_path: Path):
    samples = []
    if not txt_path.exists():
        return samples
    with open(txt_path, encoding="utf-8") as fh:
        for line in fh:
            rel = line.strip()
            if not rel:
                continue
            p = REPO_ROOT / rel
            samples.append((p, class_from_path(p)))
    return samples


def discover_samples(archive_dirs):
    samples = []
    for archive_dir in archive_dirs:
        for root, _, files in os.walk(archive_dir):
            if Path(root).name in EXCLUDE_FOLDERS:
                continue
            for fname in files:
                if Path(fname).suffix.lower() in VALID_EXTS:
                    fpath = Path(root) / fname
                    samples.append((fpath, class_from_path(fpath)))
    return samples


class BloodCellDataset(Dataset):
    def __init__(self, samples, class_to_idx, augment=False):
        self.samples      = samples
        self.class_to_idx = class_to_idx

        self.aug_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.RandomRotation(15),
        ]) if augment else None

        self.to_tensor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (64, 64), (128, 128, 128))
        if self.aug_tf:
            img = self.aug_tf(img)
        return self.to_tensor(img), self.class_to_idx[label]


PATCH_SIZE = 14

def _ceil14(n):
    return ((n + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE

def pad_collate(batch):
    tensors, labels = zip(*batch)
    max_h = _ceil14(max(t.shape[1] for t in tensors))
    max_w = _ceil14(max(t.shape[2] for t in tensors))
    padded = [F.pad(t, (0, max_w - t.shape[2], 0, max_h - t.shape[1])) for t in tensors]
    return torch.stack(padded), torch.tensor(labels, dtype=torch.long)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DinoBloomClassifier(nn.Module):
    def __init__(self, backbone, num_classes: int, feat_dim: int = 1536, unfreeze_blocks: int = 4):
        super().__init__()
        self.backbone = backbone

        for p in self.backbone.parameters():
            p.requires_grad_(False)

        blocks = list(self.backbone.blocks)
        for block in blocks[-unfreeze_blocks:]:
            for p in block.parameters():
                p.requires_grad_(True)

        if hasattr(self.backbone, "norm"):
            for p in self.backbone.norm.parameters():
                p.requires_grad_(True)

        if is_main():
            frozen  = sum(p.numel() for p in self.backbone.parameters() if not p.requires_grad)
            tunable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
            print(f"[backbone] frozen={frozen/1e6:.1f}M  trainable={tunable/1e6:.1f}M")

        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


def load_dinobloom_backbone(pth_path, device):
    sys.path.insert(0, str(REPO_ROOT))
    from dinov2.hub.backbones import dinov2_vitg14

    model = dinov2_vitg14(pretrained=False, img_size=224)
    raw   = torch.load(pth_path, map_location="cpu", weights_only=False)

    if isinstance(raw, dict):
        for key in ("teacher", "model", "state_dict"):
            if key in raw:
                raw = raw[key]
                break

    cleaned = {}
    for k, v in raw.items():
        for prefix in ("backbone.", "module.", "encoder."):
            if k.startswith(prefix):
                k = k[len(prefix):]
                break
        cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if is_main():
        print(f"[DinoBloom-G] loaded | missing={len(missing)}  unexpected={len(unexpected)}")
    return model.to(device)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    if is_main():
        print(f"World size : {dist.get_world_size()} GPUs")
        print(f"GPU 0      : {torch.cuda.get_device_name(0)}")

    # ── Data ──────────────────────────────────────────────────────────────
    all_samples = []

    txt_train = load_txt_samples(REPO_ROOT / "New Data" / "train.txt")
    txt_val   = load_txt_samples(REPO_ROOT / "New Data" / "val.txt")
    all_samples.extend(txt_train)
    all_samples.extend(txt_val)

    if is_main():
        print(f"[txt] archive5 {len(txt_train)} train + {len(txt_val)} val")

    extracted_root = REPO_ROOT / "New Data" / "extracted"
    for name in ["archive6", "archive7", "archive8"]:
        d = extracted_root / name
        if d.exists():
            found = discover_samples([d])
            if is_main():
                print(f"[dir] {name} → {len(found)} images")
            all_samples.extend(found)

    if not all_samples:
        raise RuntimeError("No training samples found.")

    all_labels   = sorted({lbl for _, lbl in all_samples})
    class_to_idx = {c: i for i, c in enumerate(all_labels)}
    num_classes  = len(class_to_idx)

    random.seed(42)
    random.shuffle(all_samples)
    test_split    = int(len(all_samples) * 0.1)
    test_samples  = all_samples[:test_split]
    train_samples = all_samples[test_split:]

    if is_main():
        print(f"\nClasses ({num_classes}): {all_labels}")
        print(f"Train : {len(train_samples)}  |  Test : {len(test_samples)}\n")
        with open(REPO_ROOT / "class_mapping.json", "w") as fh:
            json.dump({"class_to_idx": class_to_idx,
                       "idx_to_class": {v: k for k, v in class_to_idx.items()}}, fh, indent=2)

    train_ds = BloodCellDataset(train_samples, class_to_idx, augment=True)
    test_ds  = BloodCellDataset(test_samples,  class_to_idx, augment=False)

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    test_sampler  = DistributedSampler(test_ds,  shuffle=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=(args.workers > 0),
        collate_fn=pad_collate, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, sampler=test_sampler,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=(args.workers > 0),
        collate_fn=pad_collate,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    if is_main():
        print(f"Loading DinoBloom-G (unfreezing last {args.unfreeze_blocks} blocks)...")
    backbone = load_dinobloom_backbone(str(REPO_ROOT / "DinoBloom-G.pth"), device)
    model    = DinoBloomClassifier(backbone, num_classes,
                                   unfreeze_blocks=args.unfreeze_blocks).to(device)
    model    = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # ── Optimiser ─────────────────────────────────────────────────────────
    backbone_params = [p for p in model.module.backbone.parameters() if p.requires_grad]
    head_params     = list(model.module.head.parameters())

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params,     "lr": args.lr},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda")
    ce_fn  = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0
    start_epoch  = 1
    out_path     = REPO_ROOT / "bloom_leukemia.pth"
    ckpt_path    = REPO_ROOT / "checkpoint_latest.pth"
    backup_dir   = REPO_ROOT / "backup"
    best_oracle_model = None
    if is_main():
        backup_dir.mkdir(exist_ok=True)
        metrics_path = REPO_ROOT / "training_metrics.csv"
        if not metrics_path.exists():
            with open(metrics_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "train_acc", "test_acc", "lr", "best_test_acc", "timestamp"])
    else:
        metrics_path = None

    # ── Resume ────────────────────────────────────────────────────────────
    if args.resume and ckpt_path.exists():
        map_loc = {"cuda:0": f"cuda:{local_rank}"}
        ckpt = torch.load(str(ckpt_path), map_location=map_loc, weights_only=False)
        model.module.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch  = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", ckpt.get("best_train_acc", 0.0))
        if is_main():
            print(f"[resume] Resuming from epoch {start_epoch} (best val: {best_val_acc:.1f}%)\n")

    # ── Epoch loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        run_loss = 0.0
        correct  = total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                logits = model(imgs)
                loss   = ce_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            run_loss += loss.item()
            correct  += (logits.argmax(1) == labels).sum().item()
            total    += labels.size(0)

        scheduler.step()
        train_acc  = correct / total * 100
        train_loss = run_loss / len(train_loader)

        # ── Eval (all ranks) ──────────────────────────────────────────────
        model.eval()
        test_correct = test_total = 0
        class_correct = [0] * num_classes
        class_total   = [0] * num_classes

        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.amp.autocast("cuda"):
                    logits = model(imgs)
                preds = logits.argmax(1)
                test_correct += (preds == labels).sum().item()
                test_total   += labels.size(0)
                for p, l in zip(preds, labels):
                    class_correct[l.item()] += (p == l).item()
                    class_total[l.item()]   += 1

        # Aggregate across all GPUs
        stats = torch.tensor([test_correct, test_total], dtype=torch.float32, device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        test_acc = (stats[0] / stats[1] * 100).item()

        # ── Save & print (rank 0 only) ─────────────────────────────────────
        if is_main():
            idx_to_class = {v: k for k, v in class_to_idx.items()}
            print(f"\n{'='*60}")
            print(f"  Epoch {epoch:3d} / {args.epochs}")
            print(f"{'='*60}")
            print(f"  Loss      : {train_loss:.4f}")
            print(f"  Train Acc : {train_acc:.2f}%")
            print(f"  Test  Acc : {test_acc:.2f}%")
            print(f"  LR        : {scheduler.get_last_lr()[0]:.2e}")
            for i in range(num_classes):
                if class_total[i] > 0:
                    acc = class_correct[i] / class_total[i] * 100
                    print(f"    {idx_to_class[i]:<20} {acc:.1f}%  ({class_correct[i]}/{class_total[i]})")
            print(f"{'='*60}\n")

            with open(metrics_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, round(train_loss, 4), round(train_acc, 2),
                                 round(test_acc, 2), f"{scheduler.get_last_lr()[0]:.2e}",
                                 round(best_val_acc, 2), time.strftime("%Y-%m-%d %H:%M:%S")])

            ckpt_data = {
                "epoch"               : epoch,
                "best_train_acc"      : best_val_acc,
                "model_state_dict"    : model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict"   : scaler.state_dict(),
                "num_classes"         : num_classes,
                "class_to_idx"        : class_to_idx,
            }
            torch.save(ckpt_data, str(ckpt_path))

            # Upload per-epoch backup to Oracle then delete local copy — no disk buildup
            backup_path = backup_dir / f"checkpoint_epoch_{epoch:03d}.pth"
            torch.save(ckpt_data, str(backup_path))
            try:
                oracle_upload(str(backup_path), f"{OCI_RUN_PREFIX}/backups/checkpoint_epoch_{epoch:03d}.pth")
                backup_path.unlink()  # delete local copy after successful upload
                print(f"  [backup] epoch {epoch:03d} → Oracle, local copy removed")
            except Exception as e:
                print(f"  [backup] WARNING: epoch backup upload failed — keeping local — {e}")

            # Upload last checkpoint to Oracle (overwrites previous last every epoch)
            try:
                oracle_upload(str(ckpt_path), f"{OCI_RUN_PREFIX}/last.pth")
            except Exception as e:
                print(f"  [oracle] WARNING: last upload failed — {e}")

            if test_acc > best_val_acc:
                best_val_acc = test_acc
                torch.save({
                    "epoch"           : epoch,
                    "train_acc"       : round(train_acc, 4),
                    "test_acc"        : round(test_acc, 4),
                    "num_classes"     : num_classes,
                    "class_to_idx"    : class_to_idx,
                    "arch"            : "bloom_leukemia",
                    "unfreeze_blocks" : args.unfreeze_blocks,
                    "model_state_dict": model.module.state_dict(),
                }, str(out_path))
                print(f"  *** New best model — test_acc={test_acc:.2f}% — uploading to Oracle ***")
                try:
                    oracle_upload(str(out_path), f"{OCI_RUN_PREFIX}/best.pth")
                    best_oracle_model = f"{OCI_RUN_PREFIX}/best.pth"
                except Exception as e:
                    print(f"  [oracle] WARNING: best upload failed — {e}")

        dist.barrier()

    if is_main():
        print(f"\nDone. Best test acc: {best_val_acc:.2f}%")
        print(f"Model: {out_path}")

    cleanup_ddp()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",          type=int,   default=30)
    parser.add_argument("--batch-size",      type=int,   default=8,
                        help="Per-GPU batch size. Effective batch = batch_size * num_gpus")
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--unfreeze-blocks", type=int,   default=4)
    parser.add_argument("--workers",         type=int,   default=8)
    parser.add_argument("--resume",          action="store_true")
    parser.add_argument("--report-every",   type=int,   default=5)
    args = parser.parse_args()
    train(args)

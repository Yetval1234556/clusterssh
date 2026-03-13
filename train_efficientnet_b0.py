#!/usr/bin/env python3
"""
Fine-tunes DinoBloom-G.pth directly on blood cell data.

Architecture:
  DinoBloom-G (ViT-G/14)
    - Early blocks: FROZEN  (already know blood cells)
    - Last N blocks: TRAINABLE  (adapt to your classes)
  + Classification head (MLP): TRAINABLE

Output: dinobloom_g_finetuned.pth
  → The full DinoBloom-G model with your fine-tuned layers and head baked in.

Data sources:
  - New Data/train.txt + val.txt  (archive5, pre-split)
  - New Data/extracted/archive6, archive7, archive8  (auto-discovered, 80/20 split)

Images kept at original resolution. Batches padded to largest image in batch.
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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import random
import warnings
from epoch_report import EpochReporter

REPO_ROOT     = Path(__file__).parent.resolve()
OCI_NAMESPACE = "idcsxwupyymi"
OCI_BUCKET    = "bloomi-training-data"

# Organize bucket by environment and SLURM job ID
_run_date  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
_slurm_job = os.environ.get("SLURM_JOB_ID", "")
if _slurm_job:
    OCI_RUN_PREFIX = f"trained-models/unc-h200/job{_slurm_job}_{_run_date}"
else:
    OCI_RUN_PREFIX = f"trained-models/oracle-a100/{_run_date}"

def oracle_upload(local_path: str, object_name: str):
    """Upload a file to Oracle Object Storage."""
    subprocess.run([
        "oci", "os", "object", "put",
        "--namespace", OCI_NAMESPACE,
        "--bucket-name", OCI_BUCKET,
        "--name", object_name,
        "--file", local_path,
        "--force"
    ], check=True)
    print(f"  [oracle] uploaded → {OCI_BUCKET}/{object_name}")

def oracle_delete(object_name: str):
    """Delete an object from Oracle Object Storage."""
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

# Folder names that are data-split dirs, not cell-type class labels — skip them
EXCLUDE_FOLDERS = {
    "C-NMC_test_final_phase_data",
    "C-NMC_test_prelim_phase_data",
    "testing_data",
    "training_data",
    "extracted",
}

# Suppress xFormers "not available" warnings — model works fine without it
warnings.filterwarnings("ignore", message="xFormers is not available")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

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
            samples.append((p, p.parent.name))
    return samples


def discover_samples(archive_dirs):
    samples = []
    for archive_dir in archive_dirs:
        for root, _, files in os.walk(archive_dir):
            label = Path(root).name
            if label in EXCLUDE_FOLDERS:
                continue
            for fname in files:
                if Path(fname).suffix.lower() in VALID_EXTS:
                    samples.append((Path(root) / fname, label))
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

        # Resize to 224x224 — DinoBloom-G was trained at this size.
        # ViT attention is quadratic in patch count so larger sizes OOM fast.
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


PATCH_SIZE = 14  # DinoBloom-G patch size — dimensions must be multiples of this

def _ceil14(n):
    """Round n up to the nearest multiple of PATCH_SIZE."""
    return ((n + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE

def pad_collate(batch):
    tensors, labels = zip(*batch)
    # Round max dims up to nearest multiple of 14
    max_h = _ceil14(max(t.shape[1] for t in tensors))
    max_w = _ceil14(max(t.shape[2] for t in tensors))
    padded = [F.pad(t, (0, max_w - t.shape[2], 0, max_h - t.shape[1])) for t in tensors]
    return torch.stack(padded), torch.tensor(labels, dtype=torch.long)


# ---------------------------------------------------------------------------
# DinoBloom-G with classification head
# ---------------------------------------------------------------------------

class DinoBloomClassifier(nn.Module):
    """
    DinoBloom-G backbone + MLP classification head.
    The last `unfreeze_blocks` transformer blocks are trainable.
    Everything else is frozen.
    """
    def __init__(self, backbone, num_classes: int, feat_dim: int = 1536, unfreeze_blocks: int = 4):
        super().__init__()
        self.backbone = backbone

        # Freeze everything first
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        # Unfreeze the last N transformer blocks
        blocks = list(self.backbone.blocks)
        for block in blocks[-unfreeze_blocks:]:
            for p in block.parameters():
                p.requires_grad_(True)

        # Always unfreeze the final norm layer
        if hasattr(self.backbone, "norm"):
            for p in self.backbone.norm.parameters():
                p.requires_grad_(True)

        frozen  = sum(p.numel() for p in self.backbone.parameters() if not p.requires_grad)
        tunable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        print(f"[backbone] frozen={frozen/1e6:.1f}M  trainable={tunable/1e6:.1f}M")

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        # ViT-G returns CLS token as the image representation
        features = self.backbone(x)          # (B, feat_dim)
        return self.head(features)


# ---------------------------------------------------------------------------
# Load DinoBloom-G backbone
# ---------------------------------------------------------------------------

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
    print(f"[DinoBloom-G] loaded | missing={len(missing)}  unexpected={len(unexpected)}")
    return model.to(device)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Data — ALL images go into training, nothing held out ──────────────
    all_samples = []

    # archive5: train.txt + val.txt both used for training
    txt_train = load_txt_samples(REPO_ROOT / "New Data" / "train.txt")
    txt_val   = load_txt_samples(REPO_ROOT / "New Data" / "val.txt")
    all_samples.extend(txt_train)
    all_samples.extend(txt_val)
    print(f"\n[txt]  archive5  {len(txt_train)} train + {len(txt_val)} val = {len(txt_train)+len(txt_val)} total")

    # archive6, 7, 8: every image used for training
    extracted_root = REPO_ROOT / "New Data" / "extracted"
    for name in ["archive6", "archive7", "archive8"]:
        d = extracted_root / name
        if d.exists():
            found = discover_samples([d])
            print(f"[dir]  {name}  →  {len(found)} images")
            all_samples.extend(found)
        else:
            print(f"[dir]  {name}  →  not found (skipping)")

    if not all_samples:
        raise RuntimeError("No training samples found.")

    all_labels   = sorted({lbl for _, lbl in all_samples})
    class_to_idx = {c: i for i, c in enumerate(all_labels)}
    num_classes  = len(class_to_idx)

    # Hold out 10% as a test set for per-epoch evaluation
    random.seed(42)
    random.shuffle(all_samples)
    test_split   = int(len(all_samples) * 0.1)
    test_samples  = all_samples[:test_split]
    train_samples = all_samples[test_split:]

    print(f"\nClasses ({num_classes}): {all_labels}")
    print(f"Train : {len(train_samples)} images")
    print(f"Test  : {len(test_samples)} images (10% held out for evaluation)\n")

    with open(REPO_ROOT / "class_mapping.json", "w") as fh:
        json.dump(
            {"class_to_idx": class_to_idx,
             "idx_to_class": {v: k for k, v in class_to_idx.items()}},
            fh, indent=2,
        )

    train_ds = BloodCellDataset(train_samples, class_to_idx, augment=True)
    test_ds  = BloodCellDataset(test_samples,  class_to_idx, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=(args.workers > 0),
        collate_fn=pad_collate, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=(args.workers > 0),
        collate_fn=pad_collate,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    print(f"Loading DinoBloom-G  (unfreezing last {args.unfreeze_blocks} blocks)...")
    backbone = load_dinobloom_backbone(str(REPO_ROOT / "DinoBloom-G.pth"), device)
    model    = DinoBloomClassifier(backbone, num_classes,
                                   unfreeze_blocks=args.unfreeze_blocks).to(device)

    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[head]  trainable params in head: "
          f"{sum(p.numel() for p in model.head.parameters())/1e6:.1f}M")
    print(f"Total trainable: {total_trainable/1e6:.1f}M\n")

    # ── Optimiser — lower LR for backbone, higher for head ────────────────
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params     = list(model.head.parameters())

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},   # fine-tune gently
        {"params": head_params,     "lr": args.lr},          # head learns faster
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda")
    ce_fn  = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc      = 0.0
    start_epoch       = 1
    out_path          = REPO_ROOT / "dinobloom_g_finetuned.pth"
    ckpt_path         = REPO_ROOT / "checkpoint_latest.pth"
    backup_dir        = REPO_ROOT / "backup"
    best_oracle_model = None   # tracks current best model object name in Oracle
    backup_dir.mkdir(exist_ok=True)

    # ── Metrics CSV ───────────────────────────────────────────────────────
    metrics_path = REPO_ROOT / "training_metrics.csv"
    if not metrics_path.exists():
        with open(metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc", "test_acc", "lr", "best_test_acc", "timestamp"])

    # ── Resume ────────────────────────────────────────────────────────────
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch  = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", ckpt.get("best_train_acc", 0.0))
        print(f"[resume] Resuming from epoch {start_epoch}  (best val: {best_val_acc:.1f}%)\n")
    elif args.resume:
        print("[resume] No checkpoint found — starting from scratch.\n")

    # ── Epoch loop ────────────────────────────────────────────────────────
    reporter = EpochReporter(report_every=args.report_every)
    for epoch in range(start_epoch, args.epochs + 1):
        reporter.epoch_start()
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

        # ── Test evaluation ───────────────────────────────────────────────
        model.eval()
        test_correct = test_total = 0
        # Per-class tracking
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

        test_acc = test_correct / test_total * 100

        # ── Print ─────────────────────────────────────────────────────────
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        print(f"\n{'='*60}")
        print(f"  Epoch {epoch:3d} / {args.epochs}")
        print(f"{'='*60}")
        print(f"  Loss        : {train_loss:.4f}")
        print(f"  Train Acc   : {train_acc:.2f}%")
        print(f"  Test  Acc   : {test_acc:.2f}%")
        print(f"  LR          : {scheduler.get_last_lr()[0]:.2e}")
        print(f"  Per-class test accuracy:")
        for i in range(num_classes):
            if class_total[i] > 0:
                acc = class_correct[i] / class_total[i] * 100
                print(f"    {idx_to_class[i]:<20} {acc:.1f}%  ({class_correct[i]}/{class_total[i]})")
        print(f"{'='*60}\n")

        # ── Verbose epoch report every N epochs ───────────────────────────
        reporter.report(
            epoch=epoch,
            total_epochs=args.epochs,
            model=model,
            optimizer=optimizer,
            train_loss=train_loss,
            val_loss=None,
            train_acc=train_acc / 100,
            val_acc=test_acc / 100,
            extra={"lr": scheduler.get_last_lr()[0]},
        )

        # ── Log metrics to CSV ────────────────────────────────────────────
        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, round(train_loss, 4), round(train_acc, 2),
                             round(test_acc, 2), f"{scheduler.get_last_lr()[0]:.2e}",
                             round(best_val_acc, 2), time.strftime("%Y-%m-%d %H:%M:%S")])

        # Save resume checkpoint every epoch
        ckpt_data = {
            "epoch"               : epoch,
            "best_train_acc"      : best_val_acc,
            "model_state_dict"    : model.state_dict(),
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

        # Save best model whenever test accuracy improves
        if test_acc > best_val_acc:
            best_val_acc = test_acc
            torch.save(
                {
                    "epoch"           : epoch,
                    "train_acc"       : round(train_acc, 4),
                    "test_acc"        : round(test_acc, 4),
                    "num_classes"     : num_classes,
                    "class_to_idx"    : class_to_idx,
                    "arch"            : "dinobloom_g_finetuned",
                    "unfreeze_blocks" : args.unfreeze_blocks,
                    "model_state_dict": model.state_dict(),
                },
                str(out_path),
            )
            print(f"  *** New best model — test_acc={test_acc:.2f}% — uploading to Oracle ***")
            new_oracle_model = f"{OCI_RUN_PREFIX}/best.pth"
            try:
                oracle_upload(str(out_path), new_oracle_model)
                best_oracle_model = new_oracle_model
            except Exception as e:
                print(f"  [oracle] WARNING: best upload failed — {e}")

    print(f"\nDone.  Best test acc: {best_val_acc:.2f}%")
    print(f"Model: {out_path}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",          type=int,   default=30)
    parser.add_argument("--batch-size",      type=int,   default=4,
                        help="Keep at 4-8 — DinoBloom-G is large")
    parser.add_argument("--lr",              type=float, default=1e-4,
                        help="Head LR. Backbone gets lr*0.1 to fine-tune gently")
    parser.add_argument("--unfreeze-blocks", type=int,   default=4,
                        help="How many of DinoBloom-G's last transformer blocks to unfreeze")
    parser.add_argument("--workers",         type=int,   default=4)
    parser.add_argument("--resume",          action="store_true")
    parser.add_argument("--report-every",   type=int,   default=5)
    args = parser.parse_args()
    train(args)

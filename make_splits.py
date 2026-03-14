#!/usr/bin/env python3
"""
make_splits.py — runs on the cluster after archive sync.

  1. Scans all archive* folders under New Data/extracted/
  2. Counts every image and shows a breakdown by archive and class
  3. If train.txt / val.txt already exist → reports their sizes and exits
  4. If missing → builds a stratified 80/20 split and writes them

Relative paths written to train.txt / val.txt are rooted at ~/bloomi/
so the training scripts can resolve them as: REPO_ROOT / rel_path
"""

import os
import random
import pathlib
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
REPO_ROOT  = pathlib.Path.home() / "bloomi"
EXTRACTED  = REPO_ROOT / "New Data" / "extracted"
OUT_DIR    = REPO_ROOT / "New Data"
VALID_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLIT_RATIO = 0.80   # 80% train, 20% val
SEED        = 42

# Folder names that are structural, not class labels — skip them
EXCLUDE = {
    "extracted",
    "training_data",
    "testing_data",
    "C-NMC_test_final_phase_data",
    "C-NMC_test_prelim_phase_data",
}


def class_from_path(path: pathlib.Path) -> str:
    """
    Extract class label using the C-NMC filename convention:
        {patient_id}_{slide_id}_{image_id}_{CLASS}.ext
        e.g.  4_7_400_ALL.bmp  →  ALL
              UID_1_1_hem.bmp  →  hem

    Falls back to the parent folder name when the last underscore segment
    is purely numeric (image IDs like _001) or the filename has no underscores.

    For WBC data, folder names like WBC-Benign-001 or WBC-Malignant-Early-042
    are collapsed to WBC-Benign / WBC-Malignant-Early / etc. by stripping the
    trailing -NNN patient index.
    """
    stem      = path.stem                  # "4_7_400_ALL"
    parts     = stem.split("_")
    candidate = parts[-1]                  # "ALL"
    if candidate and any(c.isalpha() for c in candidate):
        return candidate
    folder = path.parent.name              # fallback: folder name
    # Collapse WBC-Benign-001 → WBC-Benign, WBC-Malignant-Early-042 → WBC-Malignant-Early
    import re as _re
    folder = _re.sub(r'-\d+$', '', folder)
    return folder

SEP = "─" * 70

# ── Discover archives ─────────────────────────────────────────────────────────
print()
print(SEP)
print("  BLOOM — Image Inventory & Split Generator")
print(SEP)

if not EXTRACTED.exists():
    print(f"  NOTE: {EXTRACTED} does not exist yet.")
    print("  No images found — skipping split generation.")
    print("  Place dataset images under New Data/extracted/ and re-run make_splits.py.")
    print(SEP)
    raise SystemExit(0)   # exit 0 so setup.bat does not abort

# Scan ALL subdirectories of extracted/ (archive5, archive6, LeukemiaAttri_Dataset, etc.)
archives = sorted(
    [d for d in EXTRACTED.iterdir() if d.is_dir()],
    key=lambda p: p.name,
)

if not archives:
    # No subdirs — try images directly in extracted/
    any_images = any(
        f.suffix.lower() in VALID_EXTS
        for f in EXTRACTED.rglob("*") if f.is_file()
    )
    if any_images:
        print(f"  NOTE: No subdirectories found. Scanning extracted/ directly.")
        archives = [EXTRACTED]
    else:
        print()
        print("  NOTE: No images found in New Data/extracted/")
        print(f"  Searched: {EXTRACTED}")
        print("  Skipping split generation — train.txt / val.txt not written.")
        print(SEP)
        raise SystemExit(0)

print(f"  Scanning: {[a.name for a in archives]}")
print()

# ── Scan images ───────────────────────────────────────────────────────────────
# by_class[class_label] = [rel_path, rel_path, ...]
by_class     = defaultdict(list)
archive_totals = {}

for archive in archives:
    count = 0
    for root, _, files in os.walk(archive):
        folder_label = pathlib.Path(root).name
        if folder_label in EXCLUDE:
            continue
        for fname in files:
            if pathlib.Path(fname).suffix.lower() in VALID_EXTS:
                abs_path = pathlib.Path(root) / fname
                rel_path = str(abs_path.relative_to(REPO_ROOT))
                label    = class_from_path(abs_path)
                by_class[label].append(rel_path)
                count += 1
    archive_totals[archive.name] = count

# ── Print inventory ───────────────────────────────────────────────────────────
total_images = sum(len(v) for v in by_class.values())

print(f"  {'Archive':<15}  {'Images':>10}")
print(f"  {'─'*15}  {'─'*10}")
for archive in archives:
    print(f"  {archive.name:<15}  {archive_totals[archive.name]:>10,}")
print(f"  {'─'*15}  {'─'*10}")
print(f"  {'TOTAL':<15}  {total_images:>10,}")
print()

print(f"  {'Class':<30}  {'Images':>10}")
print(f"  {'─'*30}  {'─'*10}")
for cls in sorted(by_class):
    print(f"  {cls:<30}  {len(by_class[cls]):>10,}")
print()

# ── Check if splits already exist ────────────────────────────────────────────
train_path = OUT_DIR / "train.txt"
val_path   = OUT_DIR / "val.txt"

if train_path.exists() and val_path.exists():
    with open(train_path) as f:
        n_train = sum(1 for l in f if l.strip())
    with open(val_path) as f:
        n_val = sum(1 for l in f if l.strip())
    print(SEP)
    print("  train.txt and val.txt already exist — skipping generation.")
    print(f"  train.txt : {n_train:,} images")
    print(f"  val.txt   : {n_val:,} images")
    print(SEP)
    print()
    raise SystemExit(0)

# ── Generate stratified 80/20 split ──────────────────────────────────────────
print(SEP)
print("  train.txt / val.txt not found — generating stratified 80/20 split...")
print(SEP)
print()

random.seed(SEED)
train_lines = []
val_lines   = []

print(f"  {'Class':<30}  {'Train':>8}  {'Val':>8}")
print(f"  {'─'*30}  {'─'*8}  {'─'*8}")

for cls in sorted(by_class):
    paths = by_class[cls][:]
    random.shuffle(paths)
    split      = max(1, int(len(paths) * SPLIT_RATIO))
    cls_train  = paths[:split]
    cls_val    = paths[split:]
    train_lines.extend(cls_train)
    val_lines.extend(cls_val)
    print(f"  {cls:<30}  {len(cls_train):>8,}  {len(cls_val):>8,}")

print(f"  {'─'*30}  {'─'*8}  {'─'*8}")
print(f"  {'TOTAL':<30}  {len(train_lines):>8,}  {len(val_lines):>8,}")
print()

random.shuffle(train_lines)
random.shuffle(val_lines)

OUT_DIR.mkdir(exist_ok=True)

with open(train_path, "w") as f:
    f.write("\n".join(train_lines) + "\n")

with open(val_path, "w") as f:
    f.write("\n".join(val_lines) + "\n")

print(f"  Written → {train_path}  ({len(train_lines):,} lines)")
print(f"  Written → {val_path}   ({len(val_lines):,} lines)")
print()
print(SEP)
print("  Split generation complete.")
print(SEP)
print()

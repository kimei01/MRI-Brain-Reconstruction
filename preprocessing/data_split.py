import json
import os
import random
import shutil
from pathlib import Path

import h5py

SOURCE_PATH = Path("datasets/archive")
DEST_ROOT = Path("datasets/dataset")
SPLIT_PATH = Path("splits")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
SEED = 42


def identify_sequence(filename):
    """Identify sequence type from filename."""
    filename = filename.upper()
    if "AXT2" in filename:
        return "T2"
    return None


def is_eligible_volume(file_path):
    """
    Check if a raw FastMRI volume meets the project's eligibility criteria:
    - K-space spatial dimensions: (768, 396)
    - Reconstruction RSS dimensions: (384, 384)
    - Slice counts match between k-space and RSS reference
    - Coil count is at least 4
    """
    expected_kspace_dims = (768, 396)
    expected_reference_dims = (384, 384)
    try:
        with h5py.File(file_path, "r") as sample:
            if "kspace" not in sample or "reconstruction_rss" not in sample:
                return False
            kspace_shape = sample["kspace"].shape
            reference_shape = sample["reconstruction_rss"].shape
            if kspace_shape[2:] != expected_kspace_dims:
                return False
            if reference_shape[1:] != expected_reference_dims:
                return False
            if kspace_shape[0] != reference_shape[0]:
                return False
            if kspace_shape[1] < 4:
                return False
            return True
    except Exception:
        return False


def split_files(files):
    """Partition files into train, val, and test splits."""
    random.seed(SEED)
    shuffled_files = list(files)
    random.shuffle(shuffled_files)

    total = len(shuffled_files)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    return {
        "train": shuffled_files[:train_end],
        "val": shuffled_files[train_end:val_end],
        "test": shuffled_files[val_end:]
    }


def prepare_raw_dataset():
    """Scan archive, filter eligible AXT2 volumes, partition, and copy to dataset splits."""
    # Ensure destination directories exist
    for split in ["train", "val", "test"]:
        (DEST_ROOT / split).mkdir(parents=True, exist_ok=True)
    SPLIT_PATH.mkdir(parents=True, exist_ok=True)

    print(f"Scanning for AXT2 files in {SOURCE_PATH}...")
    eligible_files = []
    for filename in sorted(os.listdir(SOURCE_PATH)):
        if not filename.endswith(".h5"):
            continue
        if identify_sequence(filename) != "T2":
            continue
        filepath = SOURCE_PATH / filename
        if is_eligible_volume(filepath):
            eligible_files.append(filepath)

    print(f"Found {len(eligible_files)} eligible AXT2 volumes.")

    splits = split_files(eligible_files)
    manifest = {}

    for split_name, file_paths in splits.items():
        dest_dir = DEST_ROOT / split_name
        manifest[split_name] = []

        print(f"\nProcessing {split_name} split ({len(file_paths)} volumes)...")
        for src_path in file_paths:
            dest_path = dest_dir / src_path.name
            manifest[split_name].append(str(dest_path.resolve()))

            if dest_path.exists():
                print(f"  Skipping existing file: {dest_path.name}")
            else:
                print(f"  Copying: {src_path.name} -> {dest_dir}")
                shutil.copy2(src_path, dest_path)

    # Save to T2_raw.json (never overwrite T2.json reserved for compressed data)
    output_manifest_path = SPLIT_PATH / "T2_raw.json"
    with open(output_manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)

    print(f"\nSaved raw split manifest to {output_manifest_path}")
    for name, file_list in manifest.items():
        print(f"  {name}: {len(file_list)} files")


if __name__ == "__main__":
    prepare_raw_dataset()

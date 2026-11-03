
import os
import json
import random

DATASET_PATH = "" 
SPLIT_PATH = ""

SEQUENCES = ["FLAIR", "T1", "T1POST", "T1PRE", "T2"]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
SEED = 42


def identify_sequence(filename):
    filename = filename.upper()

    if "AXT1POST" in filename:
        return "T1POST"
    elif "AXT1PRE" in filename:
        return "T1PRE"
    elif "AXFLAIR" in filename:
        return "FLAIR"
    elif "AXT1" in filename:
        return "T1"
    elif "AXT2" in filename:
        return "T2"

    return None


def split_files(files):
    random.shuffle(files)

    total = len(files)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    return {
        "train": files[:train_end],
        "val": files[train_end:val_end],
        "test": files[val_end:]
    }


def create_splits():
    random.seed(SEED)
    os.makedirs(SPLIT_PATH, exist_ok=True)

    sequence_files = {seq: [] for seq in SEQUENCES}

    for filename in os.listdir(DATASET_PATH):
        if not filename.endswith(".h5"):
            continue

        sequence = identify_sequence(filename)

        if sequence:
            filepath = os.path.join(DATASET_PATH, filename)
            sequence_files[sequence].append(filepath)

    for sequence, files in sequence_files.items():
        splits = split_files(files)

        output_path = os.path.join(
            SPLIT_PATH, f"{sequence}.json"
        )

        with open(output_path, "w") as f:
            json.dump(splits, f, indent=4)

        print(f"\n{sequence}")
        for name, file_list in splits.items():
            print(f"{name}: {len(file_list)} files")


if __name__ == "__main__":
    create_splits()

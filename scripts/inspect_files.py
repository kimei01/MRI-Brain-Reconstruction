
from pathlib import Path
import h5py
from collections import defaultdict

#Counts Files and slices in dataset
data_dir = Path("datasets/multicoil_train")
files = list(data_dir.glob("*.h5"))

print(f"Found {len(files)} .h5 files", flush=True)

counts = defaultdict(lambda: {"files": 0, "slices": 0})

for i, path in enumerate(files, start=1):
    with h5py.File(path, "r") as f:
        acquisition = f.attrs.get("acquisition", "unknown")

        if isinstance(acquisition, bytes):
            acquisition = acquisition.decode()

        counts[str(acquisition)]["files"] += 1
        counts[str(acquisition)]["slices"] += f["kspace"].shape[0]

    if i % 10 == 0 or i == len(files):
        print(f"Scanned {i}/{len(files)} files", flush=True)

print("\n=== Dataset Summary ===")

for acquisition, stats in sorted(counts.items()):
    print(
        f"{acquisition}: {stats['files']} files, "
        f"{stats['slices']} slices"
    )

print(
    f"\nTOTAL: {sum(x['files'] for x in counts.values())} files, "
    f"{sum(x['slices'] for x in counts.values())} slices"
)

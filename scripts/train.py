
from torch.utils.data import DataLoader
from scripts.data_loader import LazyMRIDataset

sequence = "T2"
manifest_path = f"splits/{sequence}.json"

# Initialize lazy datasets
train_dataset = LazyMRIDataset(manifest_path, "train")
val_dataset = LazyMRIDataset(manifest_path, "val")
test_dataset = LazyMRIDataset(manifest_path, "test")

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0
)

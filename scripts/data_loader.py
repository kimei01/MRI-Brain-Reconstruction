import json
import torch
import h5py
from torch.utils.data import Dataset, DataLoader


class MRIDataset(Dataset):

    def __init__(self, x_complex, x_ri, y, scale):

        self.x_complex = x_complex
        self.x_ri = x_ri
        self.y = y
        self.scale = scale

        assert len(x_complex) == len(x_ri) == len(y) == len(scale)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):

        return {
            "x_complex": self.x_complex[index],
            "x_ri": self.x_ri[index],
            "y": self.y[index],
            "scale": self.scale[index]
        }


def create_dataloaders(
    train_data,
    val_data,
    test_data,
    batch_size
):

    train_dataset = MRIDataset(*train_data)
    val_dataset = MRIDataset(*val_data)
    test_dataset = MRIDataset(*test_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader

class LazyMRIDataset(Dataset): 
    def __init__(self, manifest_path, split_name): 
        with open(manifest_path, "r") as f:
            split_manifest = json.load(f)

        # Get the volume paths belonging to this split
        volume_paths = split_manifest['train']
        # Initialize the sample index
        self.index = []
        # Iterate through each volume
        for volume_path in volume_paths:
            with h5py.File(volume_path, "r") as f:
                num_slices = f["kspace"].shape[0]
                # Create one reference per slice
                for slice_index in range (num_slices):
                    self.index.append(volume_path, slice_index)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, index):
    
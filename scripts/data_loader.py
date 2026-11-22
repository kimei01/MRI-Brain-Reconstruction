import json
from pathlib import Path
import torch
import h5py
from torch.utils.data import Dataset, DataLoader

from preprocessing.cartesian_mask import create_cartesian_mask
from preprocessing.normalization import normalization
from preprocessing.data_representation import create_representations


class LazyMRIDataset(Dataset): 
    def __init__(self, manifest_path, split_name): 
        with open(manifest_path, "r") as f:
            split_manifest = json.load(f)

        # Get the volume paths belonging to this split
        volume_paths = split_manifest[split_name]
        # Initialize the sample index
        self.index = []
        # Iterate through each volume
        for volume_path in volume_paths:
            with h5py.File(volume_path, "r") as f:
                num_slices = f["kspace"].shape[0]
                # Create one reference per slice
                for slice_index in range (num_slices):
                    self.index.append((volume_path, slice_index))
    def __len__(self):
        return len(self.index)
    def __getitem__(self, index):
        volume_path, slice_index = self.index[index]
        file_path = Path(volume_path)

        # Read only the requested slice from the compressed volume
        with h5py.File(file_path, "r") as f:
            kspace = torch.from_numpy(f["kspace"][slice_index])
            reference = torch.from_numpy(f["reconstruction_rss"][slice_index])

        # Apply undersampling mask
        mask = create_cartesian_mask(768, 396, acceleration=4)
        num_coils = kspace.shape[0]

        # Apply IFFT and center crop to each coil
        coil_tensors = []
        for coil_idx in range(num_coils):
            coil_data = kspace[coil_idx] * mask
            coil_data = torch.fft.fftshift(
                torch.fft.ifft2(
                    torch.fft.ifftshift(coil_data),
                    norm="ortho"
                )
            )

            # Center crop to match reference RSS dimensions
            target_h, target_w = reference.shape
            H, W = coil_data.shape
            start_h = (H - target_h) // 2
            start_w = (W - target_w) // 2
            coil_data = coil_data[
                start_h:start_h + target_h,
                start_w:start_w + target_w
            ]
            coil_tensors.append(coil_data)

        # Stack coils to form slice tensor [4, 384, 384]
        x = torch.stack(coil_tensors, dim=0)

        # Apply input-derived normalization to both input and reference
        x_norm, y_norm, scale = normalization(x, reference)

        # Create complex [4, 384, 384] and real/imaginary [8, 384, 384] representations
        x_complex, x_ri = create_representations(x_norm.unsqueeze(0))
        x_complex = x_complex.squeeze(0)
        x_ri = x_ri.squeeze(0)

        return {
            "x_complex": x_complex,
            "x_ri": x_ri,
            "y": y_norm,
            "scale": scale
        }

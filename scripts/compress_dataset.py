import os
import numpy as np
import torch 
import h5py
import json
from pathlib import Path
from preprocessing.coil_processing import fit_coil_compression, apply_coil_compression


sequence = "T2"
with open(f"splits/{sequence}.json", "r") as f:
    split_manifest = json.load(f)
output_root = Path("datasets/compressed")
for split in ["train", "val", "test"]:
    (output_root / split).mkdir(parents=True, exist_ok=True)



for split_name, volume_paths in split_manifest.items():
    output_dir = output_root / split_name

    for volume_path in volume_paths:

        output_path = output_dir / Path(volume_path).name 

        with h5py.File(volume_path, 'r') as sample:
            #Get Kspace and Reference Shape and establish kept dimensions 
            kspace_shape = sample["kspace"].shape
            reference_shape = sample["reconstruction_rss"].shape
            expected_kspace_dims = (768, 396)
            expected_reference_dims = (384, 384)
            #Check if file dimensions match expected dimensions
            if kspace_shape[2:] != expected_kspace_dims:
                continue

            if reference_shape[1:] != expected_reference_dims:
                continue

            if kspace_shape[0] != reference_shape[0]:
                continue
            if kspace_shape[1] < 4:
                continue
            #If match, get Kspace and Reference data, convert to pytorch tensor
            kspace = torch.from_numpy(sample["kspace"][:])
            reference = torch.from_numpy(sample["reconstruction_rss"][:])
            #Get compression basis of ACS and apply to Kspace only
            compression_basis = fit_coil_compression(kspace, center_fraction=0.08, num_virtual_coils=4)
            kspace = apply_coil_compression(kspace, compression_basis,)

            with h5py.File(output_path, "w") as output:
                if output_path.exists():
                    print(f"Skipping existing file: {output_path}")
                    continue
                output.create_dataset("kspace", data=kspace.numpy())
                output.create_dataset("reconstruction_rss", data=reference.numpy())
                output.create_dataset("compression_basis", data=compression_basis.numpy())
                        




    
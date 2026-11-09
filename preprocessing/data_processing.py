import os
import numpy as np
import torch 
import h5py
import json

from preprocessing.cartesian_mask import create_cartesian_mask
from preprocessing.coil_processing import fit_coil_compression, apply_coil_compression

#Testing
#sequence = "t2"
#with open(f"splits/{sequence}.json", "r") as f:
#    split = json.load(f)

def dataprocessing(file_paths): 
    mask = create_cartesian_mask(768,396, acceleration=4)
    slices_tensors= []
    reference_tensors = [] 
    for file_path in file_paths: 
        with h5py.File(file_path, 'r') as sample:

            kspace_shape = sample["kspace"].shape
            reference_shape = sample["reconstruction_rss"].shape
            expected_kspace_dims = (768, 396)
            expected_reference_dims = (384, 384)

            if kspace_shape[2:] != expected_kspace_dims:
                continue

            if reference_shape[1:] != expected_reference_dims:
                continue

            if kspace_shape[0] != reference_shape[0]:
                continue
            if kspace_shape[1] < 4:
                continue
            kspace = torch.from_numpy(sample["kspace"][:])
            reference = torch.from_numpy(sample["reconstruction_rss"][:])

            compression_basis = fit_coil_compression(kspace, center_fraction=0.08, num_virtual_coils=4)
            kspace = apply_coil_compression(kspace, compression_basis,)

            num_slices= kspace.shape[0]
            num_coils = kspace.shape[1]
        
            for slice_idx in range (num_slices): 
                coil_tensors = []
                for coil_idx in range(num_coils): 
                    coil_data = kspace[slice_idx, coil_idx, :, :]
                    undersampled_kspace = coil_data * mask
                    coil_data = torch.fft.fftshift(
                    #Apply IFFT
                    torch.fft.ifft2(
                            #ifftshift to shift center arrangment into IFFT shape
                            torch.fft.ifftshift(undersampled_kspace),
                            #orthonormal FFT convention, makes intensity scaling easier to manage
                            norm="ortho"
                        )
                    )

                    # Center crop reconstructed coil image to match RSS reference dimensions
                    target_h, target_w = reference[slice_idx].shape

                    H, W = coil_data.shape

                    start_h = (H - target_h) // 2
                    start_w = (W - target_w) // 2

                    coil_data = coil_data[
                        start_h:start_h + target_h,
                        start_w:start_w + target_w
                    ]

                    coil_tensors.append(coil_data)

                # Stack 20 coils into one slice
                slice_tensor = torch.stack(coil_tensors, dim=0)
                slices_tensors.append(slice_tensor)

                reference_tensor = reference[slice_idx]
                reference_tensors.append(reference_tensor)

    x = torch.stack(slices_tensors, dim=0)
    y = torch.stack(reference_tensors, dim = 0)
    return x, y 

#x,y = dataprocessing(split['train'])
#print(x.shape)
#print(y.shape)
#print(x.dtype)
#print(y.dtype)


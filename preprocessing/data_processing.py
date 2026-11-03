import numpy as np
import os 
import torch 
import h5py
from preprocessing.cartesian_mask import create_cartesian_mask



def dataprocessing(file_paths): 
    mask = create_cartesian_mask(640, 320, acceleration=4).numpy()
    
    for file_path in file_paths: 
        with h5py.File(file_path, 'r') as sample:
            kspace = sample["kspace"][:]
            reference = sample["reconstruction_rss"][:]

        num_slices= kspace.shape[0]
        num_coils = kspace.shape[1]

        slices_tensors= []
        reference_tensors = [] 
        
        for slice_idx in range (num_slices): 
            coil_tensors = []
            for coil_idx in range(num_coils): 
                coil_data = kspace[slice_idx, coil_idx, :, :]
                undersampled_kspace = coil_data * mask
                coil_data = np.fft.fftshift(
                #Apply IFFT
                np.fft.ifft2(
                        #ifftshift to shift center arrangment into IFFT shape
                        np.fft.ifftshift(undersampled_kspace),
                        #orthonormal FFT convention, makes intensity scaling easier to manage
                        norm="ortho"
                     )
                )

                #Fix dimension mismatch between slice and ref (640,320) - > (320,320)
                target_h, target_w = reference[slice_idx].shape

                H, W = coil_data.shape

                start_h = (H - target_h) // 2
                start_w = (W - target_w) // 2

                coil_data = coil_data[
                    start_h:start_h + target_h,
                    start_w:start_w + target_w
                ]

                coil_tensor = torch.from_numpy(coil_data)
                coil_tensors.append(coil_tensor)

            # Stack 20 coils into one slice
            slice_tensor = torch.stack(coil_tensors, dim=0)
            slices_tensors.append(slice_tensor)

            reference_tensor = torch.from_numpy(reference[slice_idx])
            reference_tensors.append(reference_tensor)

        x = torch.stack(slices_tensors, dim=0)
        y = torch.stack(reference_tensors, dim = 0)
    return x, y 

x,y = dataprocessing()
print(x.shape)
print(y.shape)
print(x.dtype)
print(y.dtype)


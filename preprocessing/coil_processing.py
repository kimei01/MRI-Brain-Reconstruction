import torch 

def fit_coil_compression(kspace, center_fraction=0.08, num_virtual_coils=4):
    if not torch.is_complex(kspace):
        raise TypeError("Expected complex-valued k-space.")

    num_slices, num_coils, height, width = kspace.shape

    #Extract the central ACS region
    num_acs = int(width * center_fraction)
    center_start = (width - num_acs) // 2
    center_end = center_start + num_acs

    acs = kspace[:, :, :, center_start:center_end]
    acs = acs.permute(0, 2, 3, 1)
    acs = acs.reshape(-1, num_coils)
    observations = acs.shape[0]

    covariance_matrix = (acs.mH @ acs) / observations
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)
    compression_basis = eigenvectors[:, -num_virtual_coils:]

    return compression_basis


def apply_coil_compression(kspace, compression_basis):

    original_shape = kspace.shape

    kspace = kspace.permute(0, 2, 3, 1)
    kspace = kspace.reshape(-1, original_shape[1])

    kspace = kspace @ compression_basis

    kspace = kspace.reshape(
        original_shape[0],
        original_shape[2],
        original_shape[3],
        compression_basis.shape[1]
    )

    kspace = kspace.permute(0, 3, 1, 2)

    return kspace

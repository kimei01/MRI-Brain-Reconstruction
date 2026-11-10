import torch

def create_representations(x):
    if not torch.is_complex(x):
        raise TypeError("Expected a complex-valued input tensor.")

    # Preserve native complex representation
    x_complex = x

    # Extract real and imaginary components
    x_real = x.real
    x_imag = x.imag

    # Stack real and imaginary components
    x_ri = torch.stack((x_real, x_imag), dim=2)

    # Merge coil and real/imaginary dimensions
    B, C, _, H, W = x_ri.shape
    x_ri = x_ri.reshape(B, C * 2, H, W)

    return x_complex, x_ri
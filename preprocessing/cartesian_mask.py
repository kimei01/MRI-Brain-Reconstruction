import torch

def create_cartesian_mask(height, width, acceleration=4, center_fraction=0.08):
    mask = torch.zeros(height, width)

    # Use ACS: Fully sampled central low-frequency region
    num_low_freqs = int(width * center_fraction)
    center_start = (width - num_low_freqs) // 2
    center_end = center_start + num_low_freqs

    mask[:, center_start:center_end] = 1

    # Periodic sampling outside the center
    mask[:, ::acceleration] = 1

    return mask
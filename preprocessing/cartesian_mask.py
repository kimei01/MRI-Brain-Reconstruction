
import torch

def create_cartesian_mask(height, width, acceleration=4, center_fraction=0.08):
    mask = torch.zeros(height, width)

    # Calculate fully sampled ACS region
    num_low_freqs = int(width * center_fraction)
    center_start = (width - num_low_freqs) // 2
    center_end = center_start + num_low_freqs

    # Fully sample the center
    mask[:, center_start:center_end] = 1

    # Calculate the total number of columns we want to sample
    target_samples = round(width / acceleration)

    # Calculate how many additional columns are needed
    additional_samples = target_samples - num_low_freqs

    # Get all column indices outside the ACS region
    outside_indices = torch.cat([
        torch.arange(0, center_start),
        torch.arange(center_end, width)
    ])

    # Evenly distribute additional samples outside ACS
    selected_positions = torch.linspace(
        0,
        len(outside_indices) - 1,
        steps=additional_samples
    ).round().long()

    selected_indices = outside_indices[selected_positions]

    # Apply additional sampling
    mask[:, selected_indices] = 1

    return mask

import torch  


def normalization(x, y):
    # Target-derived normalization: compute scale from target reconstruction y
    abs_y = torch.abs(y)

    if y.ndim == 2:
        # Single slice: y is (H, W), x is (C, H, W)
        scale = torch.quantile(abs_y.flatten(), 0.995).clamp(min=1e-8)
        x = x / scale
        y = y / scale
    else:
        # Batched slices: y is (N, H, W), x is (N, C, H, W)
        scale = torch.quantile(abs_y.flatten(start_dim=1), 0.995, dim=1).view(-1, 1, 1)
        scale = scale.clamp(min=1e-8)

        # Scale both x and y using the same scale factor
        x = x / scale.unsqueeze(1)
        y = y / scale

    return x, y, scale

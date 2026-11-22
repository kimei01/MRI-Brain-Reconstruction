import torch  


def normalization(x, y):
    # Calculate RSS independently 
    # RSS combines mag info across all virtual coils.
    if y.ndim == 2:
        # single slice: y is (H, W), x is (C, H, W)
        rss_x = torch.sqrt(torch.abs(x).square().sum(dim=0))

        #use the input RSS 99.5th percentile as the normalization scale
        # use clamp to dividing it by zero or extremely small values
        scale = torch.quantile(rss_x.flatten(), 0.995).clamp(min=1e-8)
        x = x / scale
        y = y / scale
    else:
        # sum across coils (dim=1) to calculate RSS for each slice.
        rss_x = torch.sqrt(torch.abs(x).square().sum(dim=1))
        # batched slices y is (N, H, W), x is (N, C, H, W)
        scale = torch.quantile(rss_x.flatten(start_dim=1), 0.995, dim=1).view(-1, 1, 1)
        scale = scale.clamp(min=1e-8)

        # scale both x and y using the same scale factor
        #unsqueeze since scale doesn't match x dimensions 
        x = x / scale.unsqueeze(1)
        y = y / scale

    return x, y, scale

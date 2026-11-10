import numpy as np
import torch  

def normalization(x,y):
    # Calculate scale for each slice
    scale = torch.quantile(torch.abs(x).flatten(start_dim=1), 0.995,dim=1).view(-1, 1, 1)
    scale = scale.clamp(min=1e-8)

    x = x / scale.unsqueeze(1)
    y = y / scale

    return x, y, scale
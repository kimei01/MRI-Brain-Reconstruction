import torch 
import numpy 
from skimage.metrics import structural_similarity

def mae(prediction, target): 
    num_pixels = target.numel()
    mae = torch.sum(torch.abs(prediction-target))/num_pixels
    return mae
def nmse(prediction, target): 
    denominator = torch.sum(target**2)
    if denominator == 0: 
        raise ValueError("division by 0, zero energy target")
    nmse = torch.sum((prediction - target) ** 2) / denominator
    return nmse 
def psnr(prediction, target): 
    num_pixels = target.numel()
    mse = torch.sum((prediction - target)**2)/num_pixels
    max_range = target.max() - target.min()
    if max_range == 0: 
        raise ValueError("PSNR undefined for a zero-range target")
    if mse == 0: 
        return torch.full_like(mse, float('inf'))
    psnr = 10*torch.log10(max_range**2/mse)
    return psnr
def ssim(prediction,target): 
    data_range = (target.max()-target.min()).item()
    if data_range == 0: 
        raise ValueError("SSIM undefined for a zero-range target")
    prediction = prediction.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    ssim_score = structural_similarity(
        target,
        prediction, 
        data_range=data_range
    )

    return ssim_score


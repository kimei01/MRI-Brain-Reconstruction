import torch
import argparse
import csv
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from scripts.data_loader import LazyMRIDataset
from utils.set_seed import set_seed
from models.real_vit.vit import RealVisionTransformer
from utils.metrics import mae, nmse, psnr, ssim

sequence = "T2"
manifest_path = f"splits/{sequence}.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

parser = argparse.ArgumentParser()
parser.add_argument("--model",type=str,choices=["real"], default="real")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--checkpoint", type=str,choices=["best", "final"],default="best")
parser.add_argument("--batch_size", type=int, default=4)
args = parser.parse_args()

#Generator 
set_seed(args.seed)
generator = torch.Generator()
generator=generator.manual_seed(args.seed)

#Experiment Path and Checkpoint Path
experiment_dir = Path(f"experiments/{args.model}_vit_seed{args.seed}")
if args.checkpoint == "best":
    checkpoint_path = experiment_dir / "vit_best.pth"
else:
    checkpoint_path = experiment_dir / "final_model.pth"

#Test Loader 
test_dataset = LazyMRIDataset(manifest_path, "test")
test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=0,
    generator = generator
)

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=True
)

if args.model == "real":
    model = RealVisionTransformer().to(device)
    input_key = "x_ri"


model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
mae_scores = []
nmse_scores = []
psnr_scores = []
ssim_scores = []

with torch.inference_mode():
    for batch in tqdm(test_loader, desc="Evaluating"):
        x = batch[input_key].to(device)
        target = batch["y"].to(device)
        scale = batch["scale"].to(device)
        prediction = model(x)

        prediction = prediction * scale.view(-1, 1, 1)
        target = target * scale.view(-1, 1, 1)
        for i in range(prediction.size(0)):
            slice_mae = mae(prediction[i], target[i])
            slice_nmse = nmse(prediction[i], target[i])
            slice_psnr = psnr(prediction[i], target[i])
            slice_ssim = ssim(prediction[i], target[i])
            mae_scores.append(slice_mae.item())
            nmse_scores.append(slice_nmse.item())
            psnr_scores.append(slice_psnr.item())
            ssim_scores.append(slice_ssim)
avg_mae = np.mean(mae_scores)
avg_nmse = np.mean(nmse_scores)
avg_psnr = np.mean(psnr_scores)
avg_ssim = np.mean(ssim_scores)

results = {
    "model": args.model,
    "seed": args.seed,
    "checkpoint": args.checkpoint,
    "epoch": checkpoint["epoch"],
    "test_slices": len(test_dataset),
    "MAE": avg_mae,
    "NMSE": avg_nmse,
    "PSNR": avg_psnr,
    "SSIM": avg_ssim
}

csv_path = experiment_dir / f"evaluation_{args.checkpoint}.csv"

with open(csv_path, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=results.keys())
    writer.writeheader()
    writer.writerow(results)

print(f"Evaluation results saved to {csv_path}")
        

import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from torch.utils.data import DataLoader

from scripts.data_loader import LazyMRIDataset
from utils.set_seed import set_seed
from models.real_vit.vit import RealVisionTransformer
from models.complex_vit.vit import ComplexVisionTransformer
from models.real_unet.unet import RealUNet
from models.complex_unet.unet import ComplexUNet
from utils.metrics import mae

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_slice_maes(model, loader, input_key, device):
    model.eval()
    scores = []
    scales = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating slices"):
            x = batch[input_key].to(device)
            y = batch["y"].to(device)
            scale = batch["scale"]

            prediction = model(x)

            for i in range(len(prediction)):
                scores.append(mae(prediction[i], y[i]).item())
                scales.append(scale[i].item())

    return scores, scales


def visualize_slice(model, dataset, index, input_key, model_title, output_dir, device, name):
    model.eval()

    sample = dataset[index]

    x = sample[input_key].unsqueeze(0).to(device)
    x_complex = sample["x_complex"]
    y = sample["y"]
    scale = sample["scale"].item()

    with torch.no_grad():
        prediction = model(x).squeeze(0).cpu()

    prediction = prediction * scale
    target = y * scale

    zero_filled = torch.sqrt(torch.sum(torch.abs(x_complex) ** 2, dim=0)) * scale
    error = torch.abs(prediction - target)

    images = [
        zero_filled.numpy(),
        target.numpy(),
        prediction.numpy()
    ]

    error = error.numpy()

    vmax = float(np.percentile(target.numpy(), 99.5))
    vmax = max(vmax, 1e-8)
    error_max = float(np.percentile(error, 99.5))
    error_max = max(error_max, 1e-8)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    titles = ["Zero-filled Input", "Ground Truth", model_title, "Absolute Error"]

    for i in range(3):
        axes[i].imshow(images[i], cmap="gray", vmin=0, vmax=vmax)
        axes[i].set_title(titles[i])
        axes[i].axis("off")

    axes[3].imshow(error, cmap="magma", vmin=0, vmax=error_max)
    axes[3].set_title(titles[3])
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / f"{name}_slice.png", dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["real_vit", "complex_vit", "real_unet", "complex_unet"], default="real_vit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, choices=["best", "final"], default="best")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--acceleration", type=int, choices=[4, 8], default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--embed_dim", type=int, default=None,
                        help="Token width. Default: 192 for real ViT, 138 for complex ViT")
    parser.add_argument("--mlp_dim", type=int, default=None,
                        help="Transformer MLP width. Default: 4 x embed_dim")
    args = parser.parse_args()

    is_vit = "vit" in args.model
    if is_vit:
        if args.embed_dim is None:
            args.embed_dim = 192 if "real" in args.model else 138
        if args.mlp_dim is None:
            args.mlp_dim = 4 * args.embed_dim
    else:
        args.embed_dim = None
        args.mlp_dim = None

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if is_vit:
        experiment_dir = Path(f"experiments/{args.model}_d{args.embed_dim}_mlp{args.mlp_dim}_acc{args.acceleration}_seed{args.seed}")
        if not experiment_dir.exists():
            fallback_dir = Path(f"experiments/{args.model}_acc{args.acceleration}_seed{args.seed}")
            if fallback_dir.exists():
                experiment_dir = fallback_dir
    else:
        experiment_dir = Path(f"experiments/{args.model}_acc{args.acceleration}_seed{args.seed}")

    config_path = experiment_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)

    if args.checkpoint == "best":
        checkpoint_path = experiment_dir / "best_model.pth"
        if not checkpoint_path.exists():
            checkpoint_path = experiment_dir / "vit_best.pth"
    else:
        checkpoint_path = experiment_dir / "final_model.pth"

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if args.model == "real_vit":
        model = RealVisionTransformer(embed_dim=args.embed_dim, mlp_dim=args.mlp_dim).to(device)
        input_key = "x_ri"
    elif args.model == "complex_vit":
        model = ComplexVisionTransformer(embed_dim=args.embed_dim, mlp_dim=args.mlp_dim).to(device)
        input_key = "x_complex"
    elif args.model == "real_unet":
        model = RealUNet(in_channels=8).to(device)
        input_key = "x_ri"
    elif args.model == "complex_unet":
        model = ComplexUNet(in_channels=4).to(device)
        input_key = "x_complex"

    model_titles = {
        "real_vit": "Real ViT",
        "complex_vit": "Complex ViT",
        "real_unet": "Real UNet",
        "complex_unet": "Complex UNet"
    }
    model_title = model_titles.get(args.model, args.model)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = LazyMRIDataset(
        "splits/T2.json",
        "test",
        acceleration=args.acceleration
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    scores, scales = get_slice_maes(model, loader, input_key, device)

    ranked = np.argsort(scores)

    selected = {
        "best": ranked[0],
        "median": ranked[len(ranked) // 2],
        "worst": ranked[-1]
    }

    # Add representative anatomical examples using slices 4, 8, and 12
    rep_vol = dataset.index[selected["median"]][0]
    for target_slice in [4, 8, 12]:
        match = [
            i for i, (vol, sl) in enumerate(dataset.index)
            if vol == rep_vol and sl == target_slice
        ]
        if not match:
            match = [
                i for i, (_, sl) in enumerate(dataset.index)
                if sl == target_slice
            ]
        if match:
            selected[f"slice_{target_slice}"] = match[0]

    output_dir = experiment_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / f"slice_mae_{args.checkpoint}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "global_index",
            "volume",
            "slice_in_volume",
            "mae_normalized",
            "mae_unnormalized",
            "scale"
        ])
        for i in range(len(scores)):
            vol_path, slice_in_vol = dataset.index[i]
            vol_name = Path(vol_path).name
            mae_norm = scores[i]
            scale_val = scales[i]
            mae_unnorm = mae_norm * scale_val
            writer.writerow([
                i,
                vol_name,
                slice_in_vol,
                mae_norm,
                mae_unnorm,
                scale_val
            ])

    for name, index in selected.items():
        vol_path, sl = dataset.index[int(index)]
        vol_name = Path(vol_path).name
        print(
            f"{name}: global index {index} ({vol_name}, slice {sl}), "
            f"Normalized MAE: {scores[index]:.6f}"
        )

        visualize_slice(
            model,
            dataset,
            int(index),
            input_key,
            model_title,
            output_dir,
            device,
            name
        )

    print(f"Visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()

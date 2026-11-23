import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import argparse
import json
from tqdm import tqdm
from torch.utils.data import DataLoader

from scripts.data_loader import LazyMRIDataset
from models.real_vit.vit import RealVisionTransformer
from models.complex_vit.vit import ComplexVisionTransformer
from models.real_unet.unet import RealUNet
from models.complex_unet.unet import ComplexUNet
from utils.set_seed import set_seed
from utils.experiment_logger import ExperimentLogger


def gradient_descent(model, train_loader, val_loader, input_key, num_epochs, decay_epochs, lr, logger, device, checkpoint_path, final_checkpoint_path):
    loss_fn = nn.L1Loss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    best_val_loss = float("inf")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=decay_epochs)
    for epochs in range(num_epochs):
        model.train()
        total_loss = 0
        total_samples = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epochs + 1}/{num_epochs}"):
            x = batch[input_key].to(device)
            target = batch["y"].to(device)
            model_outputs = model(x)

            loss = loss_fn(model_outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            train_loss = total_loss / total_samples

        val_loss = validate(model, val_loader, input_key, device)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_data = {
                "epoch": epochs + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss
            }
            torch.save(checkpoint_data, checkpoint_path)
            # Maintain vit_best.pth alias for backward compatibility with ViT evaluations
            vit_alias = checkpoint_path.parent / "vit_best.pth"
            if checkpoint_path.name != "vit_best.pth":
                torch.save(checkpoint_data, vit_alias)

        metrics = {
            "epochs": epochs + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": current_lr,
            "best_val_loss": best_val_loss
        }
        logger.log_epochs(metrics)
        print(f"Current Epoch: {epochs + 1}, Train Loss: {train_loss}, Val Loss: {val_loss}")

    torch.save({
        "epoch": epochs + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "final_val_loss": val_loss
    }, final_checkpoint_path)

    return best_val_loss


def validate(model, loader, input_key, device):
    model.eval()
    loss_fn = nn.L1Loss()
    total_loss = 0
    total_samples = 0

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Validating"):
            x = batch[input_key].to(device)
            target = batch["y"].to(device)

            prediction = model(x)
            loss = loss_fn(prediction, target)

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


def main():
    sequence = "T2"
    manifest_path = f"splits/{sequence}.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Training Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--decay_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--model", type=str, choices=["real_vit", "complex_vit", "real_unet", "complex_unet"], default="real_vit")
    parser.add_argument("--seed", type=int, default=42)
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

    # Generator
    set_seed(args.seed)
    generator = torch.Generator()
    generator = generator.manual_seed(args.seed)

    # Experiment Path and Checkpoint Path
    if is_vit:
        experiment_dir = Path(f"experiments/{args.model}_d{args.embed_dim}_mlp{args.mlp_dim}_acc{args.acceleration}_seed{args.seed}")
    else:
        experiment_dir = Path(f"experiments/{args.model}_acc{args.acceleration}_seed{args.seed}")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = experiment_dir / "best_model.pth"
    final_checkpoint_path = experiment_dir / "final_model.pth"

    # Epoch training logger
    logger = ExperimentLogger(
        experiment_dir / "training_log.csv"
    )

    # Initialize lazy datasets
    train_dataset = LazyMRIDataset(manifest_path, "train", acceleration=args.acceleration)
    val_dataset = LazyMRIDataset(manifest_path, "val", acceleration=args.acceleration)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=generator,
        persistent_workers=args.num_workers > 0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        generator=generator,
        persistent_workers=args.num_workers > 0
    )

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

    num_parameters = sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())
    if is_vit:
        print(f"Model: {args.model}, embed_dim={args.embed_dim}, mlp_dim={args.mlp_dim}, parameters={num_parameters:,}")
    else:
        print(f"Model: {args.model}, parameters={num_parameters:,}")

    config = {
        "model": args.model,
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "decay_epochs": args.decay_epochs,
        "acceleration": args.acceleration,
        "center_fraction": 0.08,
        "num_virtual_coils": 4,
        "num_workers": args.num_workers,
        "embed_dim": args.embed_dim,
        "mlp_dim": args.mlp_dim,
        "num_parameters": num_parameters
    }

    with open(experiment_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    best_val_loss = gradient_descent(
        model,
        train_loader,
        val_loader,
        input_key,
        num_epochs=args.epochs,
        decay_epochs=args.decay_epochs,
        lr=args.lr,
        logger=logger,
        device=device,
        checkpoint_path=checkpoint_path,
        final_checkpoint_path=final_checkpoint_path
    )

    print(f"Best validation loss: {best_val_loss}")


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path

from scripts.data_loader import LazyMRIDataset
from models.real_vit.vit import RealVisionTransformer

sequence = "T2"
manifest_path = f"splits/{sequence}.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default = 5)
parser.add_argument("--decay_epochs", type = int, default = 30)
parser.add_argument("--lr", type=float, default = 0.0001)
parser.add_argument("--batch_size", type=int, default = 4)
parser.add_argument("--model",type=str,choices=["real"], default="real")
#choices=["real", "complex"] when complex vit is done
args = parser.parse_args()

checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)
checkpoint_path = checkpoint_dir / f"{args.model}_vit_best.pth"

# Initialize lazy datasets
train_dataset = LazyMRIDataset(manifest_path, "train")
val_dataset = LazyMRIDataset(manifest_path, "val")
test_dataset = LazyMRIDataset(manifest_path, "test")

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size= args.batch_size,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=0
)


if args.model == "real":
    model = RealVisionTransformer().to(device)
    input_key = "x_ri"

#elif args.model == "complex":
    #model = ComplexVisionTransformer().to(device)
    #input_key = "x_complex"
def gradient_descent(model, train_loader, val_loader, input_key, num_epochs, decay_epochs, lr): 
    loss_fn = nn.L1Loss()

    optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=lr,
    weight_decay=1e-4
    )

    best_val_loss = float("inf")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=decay_epochs)
    for epochs in range (num_epochs): 
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
            
        val_loss = evaluate(model, val_loader, input_key)
        scheduler.step()
        if val_loss < best_val_loss:
            best_val_loss = val_loss 
            torch.save({
            "epoch": epochs + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss
            }, checkpoint_path) 
        print(f"Current Epoch: {epochs + 1}, Train Loss: {train_loss}, Val Loss: {val_loss}")
    return best_val_loss


def evaluate(model, loader, input_key):
    model.eval()
    loss_fn = nn.L1Loss()
    total_loss = 0
    total_samples = 0

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Evaluating"):

            x = batch[input_key].to(device)
            target = batch["y"].to(device)

            prediction = model(x)
            loss = loss_fn(prediction, target)

            batch_size = x.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


best_val_loss = gradient_descent(
    model,
    train_loader,
    val_loader,
    input_key,
    num_epochs=args.epochs,
    decay_epochs=args.decay_epochs,
    lr=args.lr
)

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])

test_loss = evaluate(model, test_loader, input_key)
print(f"Final Test Loss: {test_loss:.6f}")

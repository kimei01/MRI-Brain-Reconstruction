import torch
import torch.nn as nn
import argparse
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path

from scripts.data_loader import LazyMRIDataset
from models.real_vit.vit import RealVisionTransformer
from utils.set_seed import set_seed
from utils.experiment_logger import ExperimentLogger

sequence = "T2"
manifest_path = f"splits/{sequence}.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

#Training Args
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default = 5)
parser.add_argument("--decay_epochs", type = int, default = 30)
parser.add_argument("--lr", type=float, default = 0.0001)
parser.add_argument("--batch_size", type=int, default = 4)
parser.add_argument("--model",type=str,choices=["real"], default="real")
parser.add_argument("--seed", type=int, default=42)
#choices=["real", "complex"] when complex vit is done
args = parser.parse_args()

#Generator 
set_seed(args.seed)
generator = torch.Generator()
generator=generator.manual_seed(args.seed)

#Experiment Path and Checkpoint Path
experiment_dir = Path(f"experiments/{args.model}_vit_seed{args.seed}")
experiment_dir.mkdir(parents=True, exist_ok=True)
checkpoint_path = experiment_dir / "vit_best.pth"
final_checkpoint_path = experiment_dir / "final_model.pth"
config = {
        "model": args.model,
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "decay_epochs": args.decay_epochs,
    } 

with open(experiment_dir/"config.json", "w") as f: 
    json.dump(config, f, indent = 4)
    
#Epoch training logger
logger = ExperimentLogger(
    experiment_dir/"training_log.csv"
)

# Initialize lazy datasets
train_dataset = LazyMRIDataset(manifest_path, "train")
val_dataset = LazyMRIDataset(manifest_path, "val")

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size= args.batch_size,
    shuffle=True,
    num_workers=0,
    generator = generator
)

val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=0,
    generator = generator 
)

if args.model == "real":
    model = RealVisionTransformer().to(device)
    input_key = "x_ri"

#elif args.model == "complex":
    #model = ComplexVisionTransformer().to(device)
    #input_key = "x_complex"

def gradient_descent(model, train_loader, val_loader, input_key, num_epochs, decay_epochs, lr, logger): 
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
        current_lr = optimizer.param_groups[0]["lr"]
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
        "final_val_loss": val_loss}
        ,final_checkpoint_path
    )
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
    lr=args.lr,
    logger=logger
)



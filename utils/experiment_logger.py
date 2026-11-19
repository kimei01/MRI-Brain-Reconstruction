import csv
from pathlib import Path
class ExperimentLogger:
    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = [
            "epochs",
            "train_loss",
            "val_loss",
            "learning_rate",
            "best_val_loss"
        ]

        with open(self.log_path, "w", newline="") as file:
            writer = csv.DictWriter(
                file,
                self.fieldnames
            )
            writer.writeheader()
    def log_epochs(self, metrics): 
        with open(self.log_path, "a", newline="") as file:
            writer = csv.DictWriter(
                file, 
                self.fieldnames
            )
            writer.writerow(metrics)


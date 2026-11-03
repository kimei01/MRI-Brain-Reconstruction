
import torch
from torch.utils.data import Dataset, DataLoader


class MRIDataset(Dataset):

    def __init__(self, x_complex, x_ri, y):

        self.x_complex = x_complex
        self.x_ri = x_ri
        self.y = y

        assert len(x_complex) == len(x_ri) == len(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):

        return {
            "x_complex": self.x_complex[index],
            "x_ri": self.x_ri[index],
            "y": self.y[index]
        }


def create_dataloaders(
    train_data,
    val_data,
    test_data,
    batch_size
):

    train_dataset = MRIDataset(*train_data)
    val_dataset = MRIDataset(*val_data)
    test_dataset = MRIDataset(*test_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader

# dataset.py
import torch
from torch.utils.data import Dataset, DataLoader

class CropDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        """
        X: tensor [N, C=4, T=12, H=24, W=24] (ou numpy array)
        Y: tensor [N, H, W] (classe pixel-wise)
        """
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]

        if self.transform:
            x = self.transform(x)

        return x, y

def get_train_loader(X, Y, batch_size=8, shuffle=True, transform=None):
    dataset = CropDataset(X, Y, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader
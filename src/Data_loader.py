import h5py
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class CropDataset(Dataset):
    def __init__(self, h5_path, transform=None):
        """
        Charge les données depuis un fichier .h5 au format:
        - data: [N, T, C, H, W]
        - labels: [N, H, W]
        - coords: [N, 2] (optionnel)
        - dates: [T] (optionnel)

        Args:
            h5_path (str): chemin vers le fichier .h5
            transform (callable, optional): transformation à appliquer sur les patches d'entrée
        """
        with h5py.File(h5_path, 'r') as hf:
            self.X = torch.tensor(hf['dataset/data'][:])  # [N, T, C, H, W]
            self.Y = torch.tensor(hf['dataset/gt_labels'][:]).long()  # [N, H, W]
            # coords et dates sont optionnels, on peut les stocker aussi si besoin
            if 'coords' in hf['dataset']:
                self.coords = hf['dataset/coords'][:]
            else:
                self.coords = None
            if 'dates' in hf['dataset']:
                self.dates = hf['dataset/dates'][:]
            else:
                self.dates = None

        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]  # [T, C, H, W]
        y = self.Y[idx]  # [H, W]

        if self.transform:
            x = self.transform(x)

        return x, y

def get_dataloaders_from_h5(h5_path, batch_size=8, shuffle=True, transform=None, val_ratio=0.2, random_seed=42):
    dataset = CropDataset(h5_path, transform=transform)

    n_total = len(dataset)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val

    torch.manual_seed(random_seed)
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
import h5py
import torch
from torch.utils.data import Dataset, DataLoader, random_split




class CropDataset(Dataset):
    def __init__(self, h5_path, transform=None):
        with h5py.File(h5_path, 'r') as hf:
            self.X = torch.tensor(hf['data'][:])  # [N, T, C, H, W]
            self.Y = torch.tensor(hf['labels'][:]).long()  # [N, H, W]
            self.coords = hf['coords'][:] 
            self.dates = hf['dates'][:]
            self.zones = hf['zones'][:]
            self.ID_Parcelles = hf["ID_Parcelles"][:]
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]          # [T, C, H, W]
        y = self.Y[idx]          # [H, W]
        x = x.permute(1, 0, 2, 3)  # [C, T, H, W] — format attendu par ton modèle
        if self.transform:
            x = self.transform(x)
        return x, y
    
    def get_coord(self, idx):
        return self.coords[idx]
    
    def get_zone(self, idx):
        zone = self.zones[idx]
        zone = str(zone)
        return zone[2:-1]

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
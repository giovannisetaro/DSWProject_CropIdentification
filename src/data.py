import h5py
import torch
import sklearn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,Subset, DataLoader, random_split

from collections import Counter

def safe_stratify(labels):
    counts = Counter(labels)
    rare_classes = [cls for cls, count in counts.items() if count < 2]
    if rare_classes:
        print(f"⚠️ Stratification désactivée. Classes trop rares : {rare_classes}")
        return None, rare_classes
    return labels, []
    

class CropTabularDataset(Dataset):
    def __init__(self, h5_path, transform=None):
        with h5py.File(h5_path, 'r') as hf:
            self.X = hf['data'][:]  # [N, T, C, H, W]
            self.Y = hf['labels'][:]  # [N, H, W]
            self.zones = hf['zones'][:]
            self.ID_Parcelles = hf['ID_Parcelles'][:]
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx])  # [T, C, H, W]
        y = torch.tensor(self.Y[idx])  # [H, W]

        # Transform to [H*W, T*C]
        T, C, H, W = x.shape
        x = x.permute(2, 3, 0, 1).reshape(-1, T * C)  # [H*W, T*C]
        y = y.flatten()  # [H*W]

        if self.transform:
            x = self.transform(x)

        return x, y
    


class CropCnnDataset(Dataset):
    def __init__(self, h5_path, transform=None):
        with h5py.File(h5_path, 'r') as hf:
            self.X = torch.tensor(hf['data'][:])  # [N, T, C, H, W]
            self.Y = torch.tensor(hf['labels'][:]).long()  # [N, H, W]

            self.zones = hf['zones'][:]
            self.ID_Parcelles = hf['ID_Parcelles'][:]

        self.transform = transform


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].permute(1, 0, 2, 3)  # [C, T, H, W]
        y = self.Y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y




#| Split        | for :                                                                                                  
#| ------------ | -------------------------------------------------------------------------------
#| `train`      | for training    => stratify by majority class per patch !                                                                              
#| `validation` | for **evaluate hyperparamètres**, to detect **over-fitting**. **early stopping** => stratify by majority class per patch ! 
#| `test`       | for **overall evaluation e** on never seen data/ on a new spatial zone   !                                       

def get_dataset_3splits(
    h5_path,
    dataset_type="cnn",
    test_split_on="zone",        # or "random"
    val_ratio=0.1,
    test_ratio=0.2,
    batch_size=8,
    transform=None,
    seed=42 ):

    # --- Load dataset class ---
    if dataset_type == "cnn":
        dataset = CropCnnDataset(h5_path, transform=transform)
    elif dataset_type == "rf":
        dataset = CropTabularDataset(h5_path, transform=transform)
        batch_size = len(dataset)
    else:
        raise ValueError("Invalid dataset_type")

    with h5py.File(h5_path, 'r') as hf:
        zones = hf["zones"][:]
        zones = [z.decode() if isinstance(z, bytes) else str(z) for z in zones]

    indices = list(range(len(dataset)))

        # === Step 1: test split by zone ===

    if test_split_on == "zone":
        stratify_labels = zones
    elif test_split_on == "random":
        stratify_labels = None
    else:
        raise ValueError("Invalid test_split_on")

    trainval_idx, test_idx = train_test_split(
        indices,
        test_size=test_ratio,
        random_state=seed,
        stratify=stratify_labels if test_split_on != "random" else None
    )

    # === Step 2: compute majority class for trainval only ===
    majority_classes = []
    for idx in trainval_idx:
        _, y = dataset[idx]
        mode = torch.mode(y.flatten())[0].item()
        majority_classes.append(mode)

    # === Step 3: Remove rare classes from trainval ===
    from collections import Counter
    counts = Counter(majority_classes)
    rare_classes = {cls for cls, count in counts.items() if count < 2}
    if rare_classes:
        print(f"❌ Suppression des classes rares dans train/val : {rare_classes}")

    # Filtrage des indices valides
    trainval_idx_filtered = [
        idx for idx in trainval_idx
        if torch.mode(dataset[idx][1].flatten())[0].item() not in rare_classes
    ]
    majority_classes_filtered = [
        torch.mode(dataset[idx][1].flatten())[0].item()
        for idx in trainval_idx_filtered
]

    # === Step 4: final train/val split with safe stratification ===
    train_idx, val_idx = train_test_split(
        trainval_idx_filtered,
        test_size=val_ratio,
        random_state=seed,
        stratify=majority_classes_filtered
    )

    # --- Collate for tabular dataset ---
    def tabular_collate_fn(batch):
        x_list, y_list = zip(*batch)
        return torch.cat(x_list), torch.cat(y_list)

    collate_fn = tabular_collate_fn if dataset_type == "rf" else None

    # --- Create data loaders ---
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader
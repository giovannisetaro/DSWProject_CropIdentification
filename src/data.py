import h5py
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
from collections import Counter


def safe_stratify(labels):
    counts = Counter(labels)
    rare_classes = [cls for cls, count in counts.items() if count < 2]
    if rare_classes:
        print(f" Stratification désactivée. Classes trop rares : {rare_classes}")
        return None, rare_classes
    return labels, []


class CropTabularDataset(Dataset):
    def __init__(self, h5_path, transform=None):
        # Load data from HDF5 file once during initialization
        with h5py.File(h5_path, 'r') as hf:
            self.X = torch.tensor(hf['data'][:])  # shape: [N, T, C, H, W]
            self.Y = torch.tensor(hf['labels'][:])  # shape: [N, H, W]
            self.zones = hf['zones'][:]
            self.ID_Parcelles = hf['ID_Parcelles'][:]
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]  # [T, C, H, W]
        y = self.Y[idx]  # [H, W]

        T, C, H, W = x.shape
        x = x.permute(2, 3, 0, 1).reshape(-1, T * C)  # [H*W, T*C]
        y = y.flatten()  # [H*W]

        if self.transform:
            x = self.transform(x)

        return x, y


class CropCnnDataset(Dataset):
    def __init__(self, h5_path, transform=None):
        self.h5_path = h5_path
        self.transform = transform

        # Open once to get length and metadata, file handle not kept open here
        with h5py.File(self.h5_path, 'r') as hf:
            self.length = len(hf['data'])
            self.zones = hf['zones'][:]
            self.ID_Parcelles = hf['ID_Parcelles'][:]

        self.hf = None  # File handler per worker, will be lazy loaded in __getitem__

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Lazy open file per worker
        if self.hf is None:
            self.hf = h5py.File(self.h5_path, 'r')

        x = self.hf['data'][idx]      # [T, C, H, W]
        y = self.hf['labels'][idx]    # [H, W]

        x = torch.tensor(x).float().permute(1, 0, 2, 3)  # [C, T, H, W]
        y = torch.tensor(y).long()

        if self.transform:
            x = self.transform(x)

        return x, y

    def __del__(self):
        if self.hf is not None:
            self.hf.close()
            self.hf = None


def get_dataset_3splits(
    trainval_h5_path,
    test_h5_path,
    dataset_type="cnn",
    val_ratio=0.1,
    batch_size=8,
    transform=None,
    seed=42):

    if dataset_type == "cnn":
        trainval_dataset = CropCnnDataset(trainval_h5_path, transform=transform)
        test_dataset = CropCnnDataset(test_h5_path, transform=transform)
    elif dataset_type == "rf":
        trainval_dataset = CropTabularDataset(trainval_h5_path, transform=transform)
        test_dataset = CropTabularDataset(test_h5_path, transform=transform)
        batch_size = len(trainval_dataset)
    else:
        raise ValueError("Invalid dataset_type")

    with h5py.File(trainval_h5_path, 'r') as hf:
        zones = hf["zones"][:]
        zones = [z.decode() if isinstance(z, bytes) else str(z) for z in zones]

    indices = list(range(len(trainval_dataset)))

    majority_classes_dict = {}
    majority_classes = []

    # WARNING: this loop can still be slow if dataset is very large,
    # consider caching majority classes if possible
    for idx in indices:
        _, y = trainval_dataset[idx]
        y_flat = y.flatten()
        y_nonzero = y_flat[y_flat != 0]
        if len(y_nonzero) > 0:
            mode = torch.mode(y_nonzero)[0].item()
        else:
            mode = 0
        majority_classes.append(mode)
        majority_classes_dict[idx] = mode

    rare_classes = {cls for cls, count in Counter(majority_classes).items() if count < 2}
    rare_classes.add(0)

    filtered_indices = [
        idx for idx in indices
        if majority_classes_dict[idx] not in rare_classes
    ]

    filtered_classes = [
        majority_classes_dict[idx] for idx in filtered_indices
    ]

    train_idx, val_idx = train_test_split(
        filtered_indices,
        test_size=val_ratio,
        random_state=seed,
        stratify=filtered_classes
    )

    def tabular_collate_fn(batch):
        x_list, y_list = zip(*batch)
        return torch.cat(x_list), torch.cat(y_list)

    collate_fn = tabular_collate_fn if dataset_type == "rf" else None

    train_set = Subset(trainval_dataset, train_idx)
    val_set = Subset(trainval_dataset, val_idx)
    test_set = test_dataset

    return train_set, val_set, test_set

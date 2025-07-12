import h5py
import torch
import sklearn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,Subset, DataLoader, random_split
from collections import Counter
import os

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
            # Load data as torch tensors to avoid repeated conversion later
            self.X = torch.tensor(hf['data'][:])  # shape: [N, T, C, H, W]
            self.Y = torch.tensor(hf['labels'][:])  # shape: [N, H, W]
            self.zones = hf['zones'][:]
            self.ID_Parcelles = hf['ID_Parcelles'][:]
        self.transform = transform

    def __len__(self):
        # Return the total number of samples
        return len(self.X)

    def __getitem__(self, idx):
        # Retrieve sample and label by index
        x = self.X[idx]  # tensor, shape: [T, C, H, W]
        y = self.Y[idx]  # tensor, shape: [H, W]

        # Reshape input tensor for tabular model:
        # Change shape from [T, C, H, W] to [H*W, T*C]
        T, C, H, W = x.shape
        x = x.permute(2, 3, 0, 1).reshape(-1, T * C)  # flatten spatial dims, combine time and channels
        y = y.flatten()  # flatten label mask to 1D [H*W]

        # Apply optional transformation (e.g. normalization)
        if self.transform:
            x = self.transform(x)

        return x, y


class CropCnnDataset(Dataset):
    def __init__(self, h5_path, transform=None):
        # Load data and labels once during initialization from HDF5
        with h5py.File(h5_path, 'r') as hf:
            # Convert data to float tensor for CNN input
            self.X = torch.tensor(hf['data'][:]).float()  # shape: [N, T, C, H, W]
            # Convert labels to long tensor for classification loss functions
            self.Y = torch.tensor(hf['labels'][:]).long()  # shape: [N, H, W]

            self.zones = hf['zones'][:]
            self.ID_Parcelles = hf['ID_Parcelles'][:]

        self.transform = transform

    def __len__(self):
        # Return number of samples
        return len(self.X)

    def __getitem__(self, idx):
        # Permute data shape for CNN input format: [C, T, H, W]
        x = self.X[idx].permute(1, 0, 2, 3)
        y = self.Y[idx]

        # Apply optional transform (e.g. data augmentation)
        if self.transform:
            x = self.transform(x)

        return x, y





#| Split        | for :                                                                                                  
#| ------------ | -------------------------------------------------------------------------------
#| `train`      | for training    => stratify by majority class per patch !                                                                              
#| `validation` | for **evaluate hyperparamètres**, to detect **over-fitting**. **early stopping** => stratify by majority class per patch ! 
#| `test`       | for **overall evaluation e** on never seen data/ on a new spatial zone   !                                       

def get_dataset_3splits(
    trainval_h5_path,       # "data/dataset_val_train.h5"
    test_h5_path,           # "data/dataset_test.h5"
    dataset_type="cnn",
    val_ratio=0.1,
    batch_size=8,
    transform=None,
    seed=42):

    # --- Load train+val dataset ---
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

    # === Step 1: compute majority class on train+val ===
    majority_classes_dict = {}
    majority_classes = []

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

    # === Step 2: Remove rare classes ===
    rare_classes = {cls for cls, count in Counter(majority_classes).items() if count < 2}
    rare_classes.add(0)

    filtered_indices = [
        idx for idx in indices
        if majority_classes_dict[idx] not in rare_classes
    ]

    filtered_classes = [
        majority_classes_dict[idx] for idx in filtered_indices
    ]

    # === Step 3: Stratified split train/val ===
    train_idx, val_idx = train_test_split(
        filtered_indices,
        test_size=val_ratio,
        random_state=seed,
        stratify=filtered_classes
    )

    # --- Collate for tabular dataset ---
    def tabular_collate_fn(batch):
        x_list, y_list = zip(*batch)
        return torch.cat(x_list), torch.cat(y_list)

    collate_fn = tabular_collate_fn if dataset_type == "rf" else None

    # --- Create dataset subsets ---
    train_set = Subset(trainval_dataset, train_idx)
    val_set = Subset(trainval_dataset, val_idx)
    test_set = test_dataset  # Full test dataset from external file

    return train_set, val_set, test_set

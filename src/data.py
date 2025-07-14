import torch
from torch.utils.data import Dataset, Subset
import h5py
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np

class CropCnnDataset(Dataset):
    def __init__(self, h5_path, transform=None, debug=False, num_classes=50, ignore_value=255):
        """
        Lazily loads data from an HDF5 file.
        Opens the file only once per worker to avoid repeated I/O overhead.

        Args:
            h5_path (str): Path to the HDF5 file.
            transform (callable): Optional transform to apply to each input sample.
            debug (bool): Whether to print debug info on each sample.
            num_classes (int): Maximum number of valid class labels.
            ignore_value (int): Value to treat as invalid (e.g. 255).
        """
        self.h5_path = h5_path
        self.transform = transform
        self.debug = debug
        self.num_classes = num_classes
        self.ignore_value = ignore_value
        self.h5_file = None

        # Open once to determine dataset length
        with h5py.File(h5_path, 'r') as hf:
            self.length = hf['data'].shape[0]

    def __getitem__(self, idx):
        # Open the file if not already opened (one per worker)
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')

        # Load input and label
        x = torch.tensor(self.h5_file['data'][idx]).float()  # shape: [T, C, H, W]
        y = torch.tensor(self.h5_file['labels'][idx]).long() # shape: [H, W]

        # Convert abnormal values (e.g., 255) to background class (0)
        y[y == self.ignore_value] = 0  # or use a valid default class if 0 is not meaningful

        # Optional: sanity check for labels out of range
        if self.debug:
            max_label = y.max().item()
            min_label = y.min().item()
            if max_label >= self.num_classes or min_label < 0:
                print(f"[WARNING] Invalid label range at sample {idx}: min={min_label}, max={max_label}")
            else:
                print(f"[DEBUG] Sample {idx} - y.min(): {min_label}, y.max(): {max_label}")

        # Permute to [C, T, H, W] for CNN input
        x = x.permute(1, 0, 2, 3)

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return self.length

    def __del__(self):
        # Ensure the file is closed on deletion
        if self.h5_file is not None:
            try:
                self.h5_file.close()
            except:
                pass


def get_dataset_3splits(
    h5_path_trainval,
    h5_path_test,
    dataset_type="cnn",
    val_ratio=0.1,
    transform=None,
    seed=42,
    debug=False,
    num_classes=50
):
    """
    Loads training, validation, and test datasets from separate HDF5 files.
    Performs a stratified split based on majority class in each label mask.

    Args:
        h5_path_trainval (str): Path to train+val HDF5 file.
        h5_path_test (str): Path to test HDF5 file.
        dataset_type (str): Type of dataset to return ("cnn" supported).
        val_ratio (float): Fraction of trainval data to use for validation.
        transform (callable): Optional transform to apply to each sample.
        seed (int): Random seed for reproducibility.
        debug (bool): If True, prints label statistics.
        num_classes (int): Number of valid classes (for sanity checks).
    
    Returns:
        train_dataset, val_dataset, test_dataset: PyTorch Dataset or Subset instances.
    """

    # Read all labels once to enable stratified splitting
    with h5py.File(h5_path_trainval, 'r') as hf:
        labels = hf['labels'][:]  # shape: [N, H, W]

    # Compute majority (non-zero) class per sample
    majority_classes = []
    for y in labels:
        y_flat = y.flatten()
        y_nonzero = y_flat[y_flat != 0]
        y_valid = y_nonzero[y_nonzero != 255]
        if len(y_valid) > 0:
            mode_class = torch.mode(torch.tensor(y_valid))[0].item()
        else:
            mode_class = 0
        majority_classes.append(mode_class)

    # Exclude rare classes (fewer than 2 samples) and class 0 (background)
    counts = Counter(majority_classes)
    rare_classes = {cls for cls, c in counts.items() if c < 2}
    rare_classes.add(0)

    # Keep only samples with valid majority classes
    valid_indices = [i for i, cls in enumerate(majority_classes) if cls not in rare_classes]
    valid_majority_classes = [majority_classes[i] for i in valid_indices]

    # Stratified train/val split
    train_idx, val_idx = train_test_split(
        valid_indices,
        test_size=val_ratio,
        random_state=seed,
        stratify=valid_majority_classes
    )

    if dataset_type == "cnn":
            full_trainval_dataset = CropCnnDataset(h5_path_trainval, transform=transform, debug=debug, num_classes=num_classes)
            test_dataset = CropCnnDataset(h5_path_test, transform=transform, debug=debug, num_classes=num_classes)
    else:
        raise NotImplementedError(f"Dataset type '{dataset_type}' not implemented")

    train_dataset = Subset(full_trainval_dataset, train_idx)
    val_dataset = Subset(full_trainval_dataset, val_idx)

    # Debug: show label ranges in subsets
    if debug:
        print("[DEBUG] Checking label ranges in subsets:")
        train_samples = [full_trainval_dataset[i][1] for i in train_idx[:10]]
        train_all_vals = torch.cat([y.flatten() for y in train_samples])
        print(f"Train labels: min={train_all_vals.min().item()}, max={train_all_vals.max().item()}")

        val_samples = [full_trainval_dataset[i][1] for i in val_idx[:10]]
        val_all_vals = torch.cat([y.flatten() for y in val_samples])
        print(f"Val labels: min={val_all_vals.min().item()}, max={val_all_vals.max().item()}")

    return train_dataset, val_dataset, test_dataset

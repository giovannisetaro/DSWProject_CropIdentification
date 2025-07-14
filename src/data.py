import torch
from torch.utils.data import Dataset
import h5py
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from typing import Optional, Callable, Tuple

class CropCnnDataset(Dataset):
    def __init__(
        self,
        h5_path: str,
        transform: Optional[Callable] = None,
        debug: bool = False,
        num_classes: int = 51,
        ignore_value: int = 255
    ) -> None:
        """
        Dataset that lazily loads data from an HDF5 file,
        opening the file once per worker (thread-safe with swmr=True).

        Args:
            h5_path (str): Path to the HDF5 file.
            transform (callable, optional): Transform to apply to inputs.
            debug (bool): If True, print debug information.
            num_classes (int): Number of valid classes (excluding ignore_value).
            ignore_value (int): Label value used to ignore pixels (e.g., 255).
        """
        self.h5_path = h5_path
        self.transform = transform
        self.debug = debug
        self.num_classes = num_classes
        self.ignore_value = ignore_value
        self.h5_file: Optional[h5py.File] = None

        # Open temporarily to read dataset length
        with h5py.File(h5_path, 'r') as hf:
            self.length = hf['data'].shape[0]

    def _open_file(self) -> None:
        if self.h5_file is None:
            # Open the HDF5 file in read-only mode with Single Writer Multiple Reader enabled
            self.h5_file = h5py.File(self.h5_path, 'r', swmr=True)
            if self.debug:
                print(f"[DEBUG] Opened HDF5 file: {self.h5_path}")

    def _close_file(self) -> None:
        if self.h5_file is not None:
            try:
                self.h5_file.close()
                if self.debug:
                    print(f"[DEBUG] Closed HDF5 file: {self.h5_path}")
            except Exception as e:
                if self.debug:
                    print(f"[WARNING] Error closing HDF5 file: {e}")
            self.h5_file = None

    def close(self) -> None:
        """Public method to explicitly close the HDF5 file."""
        self._close_file()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self._open_file()

        # Load data and label from HDF5 file
        x_np = self.h5_file['data'][idx]      # shape: [T, C, H, W]
        y_np = self.h5_file['labels'][idx]   # shape: [H, W]

        # Convert numpy arrays to torch tensors
        x = torch.from_numpy(x_np).float()
        y = torch.from_numpy(y_np).long()

        # Optional shape validation in debug mode
        if self.debug:
            assert x.ndim == 4, f"x must be 4D, got {x.ndim}"
            assert y.ndim == 2, f"y must be 2D, got {y.ndim}"
            print(f"[DEBUG] Sample {idx}: x shape={x.shape}, y shape={y.shape}")

        # Map invalid label values (<0 or >=num_classes) to ignore_value
        invalid_mask = (y < 0) | (y >= self.num_classes)
        y[invalid_mask] = self.ignore_value

        if self.debug:
            n_ignored = torch.sum(y == self.ignore_value).item()
            min_label = y.min().item()
            max_label = y.max().item()
            print(f"[DEBUG] Sample {idx}: min_label={min_label}, max_label={max_label}, ignored_pixels={n_ignored}")

        # Permute x from [T, C, H, W] to [C, T, H, W] for CNN input
        x = x.permute(1, 0, 2, 3)

        # Apply optional transform
        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self) -> int:
        return self.length

    def __del__(self) -> None:
        # Safely close the HDF5 file when dataset object is deleted
        self._close_file()


def get_dataset_3splits(
    h5_train_val_path: str,
    h5_test_path: str,
    dataset_type: str = "cnn",
    val_ratio: float = 0.1,
    transform: Optional[Callable] = None,
    seed: int = 42
) -> Tuple[Subset, Subset, Dataset]:
    """
    Load dataset, split training+validation set stratified by zones, return train/val/test sets.

    Args:
        h5_train_val_path (str): Path to training+validation HDF5 file.
        h5_test_path (str): Path to test HDF5 file.
        dataset_type (str): Dataset type ("cnn" supported).
        val_ratio (float): Fraction of training data for validation.
        transform (callable, optional): Transform applied to inputs.
        seed (int): Random seed for reproducible splitting.

    Returns:
        train_set, val_set, test_set (Subset or Dataset)
    """
    if dataset_type != "cnn":
        raise ValueError(f"Unsupported dataset_type {dataset_type}")

    train_val_dataset = CropCnnDataset(h5_train_val_path, transform=transform)
    test_dataset = CropCnnDataset(h5_test_path, transform=transform)

    # Load zones for stratified split
    with h5py.File(h5_train_val_path, 'r') as hf:
        zones_train_val = hf["zones"][:]
        zones_train_val = [z.decode() if isinstance(z, bytes) else str(z) for z in zones_train_val]

    indices = list(range(len(train_val_dataset)))

    # Stratified train/val split based on zones
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_ratio,
        random_state=seed,
        stratify=zones_train_val
    )

    train_set = Subset(train_val_dataset, train_idx)
    val_set = Subset(train_val_dataset, val_idx)
    test_set = test_dataset

    return train_set, val_set, test_set

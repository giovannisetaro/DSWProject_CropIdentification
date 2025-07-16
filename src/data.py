import torch
from torch.utils.data import Dataset, DataLoader, Subset
import h5py
from sklearn.model_selection import train_test_split
from typing import Optional, Callable, Tuple

class CropCnnDataset(Dataset):
    # Dataset class for loading crop data from an HDF5 file.
    # Supports optional transforms and debug mode.
    # num_classes specifies the number of valid classes,
    # ignore_value is used to mask invalid labels.
    def __init__(
        self,
        h5_path: str,
        transform: Optional[Callable] = None,
        debug: bool = False,
        num_classes: int = 51,
        ignore_value: int = 255
    ) -> None:
        self.h5_path = h5_path
        self.transform = transform
        self.debug = debug
        self.num_classes = num_classes
        self.ignore_value = ignore_value

        # Open the HDF5 file to get dataset length, then close immediately.
        with h5py.File(h5_path, 'r') as hf:
            self.length = hf['data'].shape[0]

        # Placeholder for lazy file opening for multiprocessing.
        self.h5_file = None 

    # Open the HDF5 file if not already opened, supports safe multiprocessing access (swmr).
    def _open_file(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r', swmr=True)
            if self.debug:
                print(f"[DEBUG] Worker {torch.utils.data.get_worker_info()} aperto file {self.h5_path}")

    # Close the HDF5 file safely.
    def _close_file(self):
        if self.h5_file is not None:
            try:
                self.h5_file.close()
            except Exception:
                pass
            self.h5_file = None

    # Return a single sample and label tensor for the given index.
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self._open_file()

        # Load data and labels from the HDF5 file.
        x_np = self.h5_file['data'][idx]      # shape [T, C, H, W]
        y_np = self.h5_file['labels'][idx]   # shape [H, W]

        # Convert numpy arrays to PyTorch tensors.
        x = torch.from_numpy(x_np).float()
        y = torch.from_numpy(y_np).long()

        # Mask invalid labels outside the class range with ignore_value.
        invalid_mask = (y < 0) | (y >= self.num_classes)
        y[invalid_mask] = self.ignore_value

        # Permute data dimensions to [Channels, Time, Height, Width].
        x = x.permute(1, 0, 2, 3)

        # Apply optional transforms if provided.
        if self.transform:
            x = self.transform(x)

        return x, y

    # Return the length of the dataset.
    def __len__(self) -> int:
        return self.length

    # Ensure file is closed when dataset is deleted.
    def __del__(self):
        self._close_file()

from torch.utils.data import Dataset

class IndexedDataset(Dataset):
    # Wrapper dataset to provide a subset of samples via given indices.
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]

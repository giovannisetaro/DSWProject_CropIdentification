import torch
from torch.utils.data import Dataset, DataLoader, Subset
import h5py
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
        self.h5_path = h5_path
        self.transform = transform
        self.debug = debug
        self.num_classes = num_classes
        self.ignore_value = ignore_value

    
        with h5py.File(h5_path, 'r') as hf:
            self.length = hf['data'].shape[0]

        self.h5_file = None 

    def _open_file(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r', swmr=True)
            if self.debug:
                print(f"[DEBUG] Worker {torch.utils.data.get_worker_info()} aperto file {self.h5_path}")

    def _close_file(self):
        if self.h5_file is not None:
            try:
                self.h5_file.close()
            except Exception:
                pass
            self.h5_file = None

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self._open_file()

        x_np = self.h5_file['data'][idx]      # [T, C, H, W]
        y_np = self.h5_file['labels'][idx]   # [H, W]

        x = torch.from_numpy(x_np).float()
        y = torch.from_numpy(y_np).long()

        invalid_mask = (y < 0) | (y >= self.num_classes)
        y[invalid_mask] = self.ignore_value

        x = x.permute(1, 0, 2, 3)  # [C, T, H, W]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self) -> int:
        return self.length

    def __del__(self):
        self._close_file()

from torch.utils.data import Dataset

class IndexedDataset(Dataset):
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]
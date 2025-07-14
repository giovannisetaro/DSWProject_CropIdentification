import torch
from torch.utils.data import Dataset
import h5py

class CropCnnDataset(Dataset):
    def __init__(self, h5_path, transform=None, debug=False, num_classes=50, ignore_value=255):
        """
        Dataset that lazily loads data from an HDF5 file.
        Opens the file once per worker to avoid overhead of repeatedly opening it.

        Args:
            h5_path (str): Path to the HDF5 file.
            transform (callable, optional): Optional transform to apply to inputs.
            debug (bool): Whether to print debug info about labels.
            num_classes (int): Number of classes expected in labels.
            ignore_value (int): Label value to ignore and convert to background (e.g. 255).
        """
        self.h5_path = h5_path
        self.transform = transform
        self.debug = debug
        self.num_classes = num_classes
        self.ignore_value = ignore_value
        self.h5_file = None

        # Open once to get dataset length
        with h5py.File(h5_path, 'r') as hf:
            self.length = hf['data'].shape[0]

    def __getitem__(self, idx):
        # Open file on first access per worker (lazy loading)
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')

        # Load data and label tensors
        x = torch.tensor(self.h5_file['data'][idx]).float()  # shape: [T, C, H, W]
        y = torch.tensor(self.h5_file['labels'][idx]).long() # shape: [H, W]

        # Convert ignore_value labels (e.g., 255) to background (0)
        y[y == self.ignore_value] = 0

        # Optional debug: check label ranges
        if self.debug:
            max_label = y.max().item()
            min_label = y.min().item()
            if max_label >= self.num_classes or min_label < 0:
                print(f"[WARNING] Sample {idx} label out of range: min={min_label}, max={max_label}")

        # Permute input tensor to [C, T, H, W] for CNN input
        x = x.permute(1, 0, 2, 3)

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return self.length

    def __del__(self):
        # Close the HDF5 file when the dataset object is deleted
        if self.h5_file is not None:
            try:
                self.h5_file.close()
            except:
                pass

from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import h5py

def get_dataset_3splits(
    h5_train_val_path,
    h5_test_path,
    dataset_type="cnn",
    val_ratio=0.1,
    batch_size=8,
    transform=None,
    seed=42
):
    """
    Load datasets and create train/validation splits.

    Args:
        h5_train_val_path (str): Path to HDF5 file containing training+validation data.
        h5_test_path (str): Path to HDF5 file containing test data.
        dataset_type (str): Type of dataset ("cnn" supported).
        val_ratio (float): Fraction of training data to use as validation.
        batch_size (int): Batch size (not used here but could be used for custom datasets).
        transform (callable): Optional transform applied to inputs.
        seed (int): Random seed for reproducibility.

    Returns:
        train_set, val_set, test_set (torch.utils.data.Dataset or Subset): Datasets for training, validation, and testing.
    """
    if dataset_type == "cnn":
        train_val_dataset = CropCnnDataset(h5_train_val_path, transform=transform)
        test_dataset = CropCnnDataset(h5_test_path, transform=transform)
    else:
        raise ValueError("Invalid dataset_type")

    # Read zones for stratification during train/val split
    with h5py.File(h5_train_val_path, 'r') as hf:
        zones_train_val = hf["zones"][:]
        # Decode bytes to strings if needed
        zones_train_val = [z.decode() if isinstance(z, bytes) else str(z) for z in zones_train_val]

    indices_train_val = list(range(len(train_val_dataset)))

    # Stratified split on zones
    train_idx, val_idx = train_test_split(
        indices_train_val,
        test_size=val_ratio,
        random_state=seed,
        stratify=zones_train_val
    )

    # Create subset datasets for train and val splits
    train_set = Subset(train_val_dataset, train_idx)
    val_set = Subset(train_val_dataset, val_idx)

    # Test dataset is the full test set
    test_set = test_dataset

    return train_set, val_set, test_set

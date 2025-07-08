import h5py
import torch
import numpy as np
from tqdm import tqdm

# Lists to store per-pixel features and labels
features_list = []
labels_list = []

# Load the HDF5 dataset
with h5py.File("data/Dataset.h5", 'r') as hf:
    X = torch.tensor(hf['data'][:])        # [N, T, C, H, W]
    Y = torch.tensor(hf['labels'][:])      # [N, H, W]

# Iterate over each sample
for x, y in tqdm(zip(X, Y), total=len(X)):
    T, C, H, W = x.shape  # x: [T, C, H, W]

    # Rearrange to [H, W, T, C] for pixel-wise access
    x_perm = x.permute(2, 3, 0, 1)  # [H, W, T, C]
    y_flat = y.flatten()  # [H * W]

    # Flatten spatial dimensions
    x_flat = x_perm.reshape(-1, T, C)  # [H * W, T, C]

    # Convert each pixel's time series to a flat vector [T*C]
    x_flat = x_flat.reshape(-1, T * C).numpy()  # [H * W, T*C]
    y_flat = y_flat.numpy()                     # [H * W]

    features_list.append(x_flat)  # list of arrays [H*W, T*C]
    labels_list.append(y_flat)   # list of arrays [H*W]

# Concatenate all samples
X_tabular = np.concatenate(features_list, axis=0)  # [N_pixels_total, T*C]
y_tabular = np.concatenate(labels_list, axis=0)    # [N_pixels_total]

# Save the tabular dataset to a .npz file for later use
np.savez("data/tabular_pixelwise_data.npz", X=X_tabular, y=y_tabular)

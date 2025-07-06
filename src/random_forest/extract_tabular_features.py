import h5py
import torch
import numpy as np
from scipy.stats import mode
from tqdm import tqdm
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Lists to store tabular features and labels
features_list = []
labels_list = []

# For CNNs, data is naturally processed as multi-dimensional tensors ([T, C, H, W]).
# However, Random Forests require 1D feature vectors and a single label per sample.
# Here we:
# 1) spatially average each channel at each timestamp to get [T, C] features,
# 2) flatten them into a 1D vector [T*C] for RF training,
# 3) select the most common pixel label as the sample’s label.

# Load the HDF5 dataset
with h5py.File("data/Dataset.h5", 'r') as hf:
    X = torch.tensor(hf['data'][:])        # [N, T, C, H, W] satellite image time series
    Y = torch.tensor(hf['labels'][:])      # [N, H, W] crop type label maps

# Iterate over each sample in the dataset
for x, y in tqdm(zip(X, Y), total=len(X)):
    # Compute the average over spatial dimensions for each timestamp and channel
    x_avg = x.mean(dim=[2, 3])  # [T, C]
    
    # Flatten the temporal sequence into a 1D feature vector [T*C]
    features = x_avg.flatten().numpy()
    
    # Compute the most frequent label (mode) in the label map as a global label
    # We use a single label per image sample because Random Forests expect one
    # output class per input feature vector (tabular format), unlike CNNs which can
    # predict per-pixel or per-region outputs.
    label = mode(y.flatten().numpy(), keepdims=False).mode
    
    # Append to the dataset lists
    features_list.append(features)
    labels_list.append(label)

# Convert to NumPy arrays for scikit-learn
X_tabular = np.stack(features_list)
y_tabular = np.array(labels_list)

# Save the tabular dataset to a .npz file for later use
np.savez("data/tabular_data.npz", X=X_tabular, y=y_tabular)

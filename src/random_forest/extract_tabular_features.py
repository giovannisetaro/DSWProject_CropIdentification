import h5py
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def flatten_samples(X, Y):
    features_list = []
    labels_list = []
    for x, y in tqdm(zip(X, Y), total=len(X)):
        T, C, H, W = x.shape
        x_perm = x.permute(2, 3, 0, 1)  # [H, W, T, C]
        y_flat = y.flatten()
        x_flat = x_perm.reshape(-1, T * C).numpy()
        labels_list.append(y_flat.numpy())
        features_list.append(x_flat)
    X_flat = np.concatenate(features_list, axis=0)
    y_flat = np.concatenate(labels_list, axis=0)
    return X_flat, y_flat

def main():
    # 1. Load dataset
    with h5py.File("data/Dataset.h5", 'r') as hf:
        X = torch.tensor(hf['data'][:])        # [N, T, C, H, W]
        Y = torch.tensor(hf['labels'][:])      # [N, H, W]

    # 2. Split dataset into train, validation, and test sets
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X, Y, test_size=0.4, random_state=42, stratify=None)  
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp, test_size=0.5, random_state=42, stratify=None)

    print(f"Train samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    # 3. Flatten the samples separately for each split
    X_train_flat, y_train_flat = flatten_samples(X_train, Y_train)
    X_val_flat, y_val_flat = flatten_samples(X_val, Y_val)
    X_test_flat, y_test_flat = flatten_samples(X_test, Y_test)

    print(f"Train pixels: {X_train_flat.shape[0]}, classes: {np.unique(y_train_flat)}")
    print(f"Val pixels: {X_val_flat.shape[0]}, classes: {np.unique(y_val_flat)}")
    print(f"Test pixels: {X_test_flat.shape[0]}, classes: {np.unique(y_test_flat)}")

    # 4. Save the tabular data for each split
    np.savez("data/train_pixelwise.npz", X=X_train_flat, y=y_train_flat)
    np.savez("data/val_pixelwise.npz", X=X_val_flat, y=y_val_flat)
    np.savez("data/test_pixelwise.npz", X=X_test_flat, y=y_test_flat)

if __name__ == "__main__":
    main()

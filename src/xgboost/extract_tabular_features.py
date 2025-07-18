import os
import h5py
import torch
import numpy as np
from collections import defaultdict

IGNORE_LABEL = 225  # label to ignore (masked out)

def flatten_samples(X, Y):
    """
    Flattens input tensors from (N, T, C, H, W) and (N, H, W)
    to 2D pixel-wise matrices: (N·H·W, T·C) and (N·H·W,)
    """
    N, T, C, H, W = X.shape
    X = X.permute(0, 3, 4, 1, 2).reshape(N * H * W, T * C)
    Y = Y.reshape(N * H * W)
    return X.numpy(), Y.numpy()

def mask_ignore_label(X, y, ignore_value=IGNORE_LABEL):
    """
    Filters out pixels where the label is equal to `ignore_value`.
    """
    mask = y != ignore_value
    return X[mask], y[mask]

def main():
    os.makedirs("data", exist_ok=True)

    # Load train+val data
    with h5py.File("data/dataset_val_train.h5", 'r') as hf:
        X_valtrain = torch.tensor(hf['data'][:])   # (N, T, C, H, W)
        Y_valtrain = torch.tensor(hf['labels'][:]) # (N, H, W)

    # Load test data
    with h5py.File("data/dataset_test.h5", 'r') as hf:
        X_test = torch.tensor(hf['data'][:])
        Y_test = torch.tensor(hf['labels'][:])

    # Greedy class coverage for training set
    N = X_valtrain.shape[0]
    all_indices = list(range(N))
    class_to_samples = defaultdict(set)
    for i in range(N):
        unique_classes = torch.unique(Y_valtrain[i])
        for cls in unique_classes:
            class_to_samples[int(cls.item())].add(i)

    train_indices = set()
    covered_classes = set()

    while len(covered_classes) < len(class_to_samples):
        best_sample = None
        best_new_classes = set()
        for i in all_indices:
            if i in train_indices:
                continue
            sample_classes = set(torch.unique(Y_valtrain[i]).tolist())
            new_classes = sample_classes - covered_classes
            if len(new_classes) > len(best_new_classes):
                best_sample = i
                best_new_classes = new_classes
        if best_sample is None:
            break
        train_indices.add(best_sample)
        covered_classes.update(torch.unique(Y_valtrain[best_sample]).tolist())

    val_indices = [i for i in all_indices if i not in train_indices]

    print(f"Train samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")
    print(f"Test samples: {X_test.shape[0]}")

    # Select train and val sets separately
    X_train = X_valtrain[list(train_indices)]
    Y_train = Y_valtrain[list(train_indices)]
    X_val = X_valtrain[val_indices]
    Y_val = Y_valtrain[val_indices]

    # Flatten each set separately
    X_train_flat, y_train_flat = flatten_samples(X_train, Y_train)
    X_val_flat, y_val_flat = flatten_samples(X_val, Y_val)
    X_test_flat, y_test_flat = flatten_samples(X_test, Y_test)

    # Apply ignore mask on each set
    X_train_flat, y_train_flat = mask_ignore_label(X_train_flat, y_train_flat)
    X_val_flat, y_val_flat = mask_ignore_label(X_val_flat, y_val_flat)
    X_test_flat, y_test_flat = mask_ignore_label(X_test_flat, y_test_flat)

    print(f"Train pixels: {X_train_flat.shape[0]}, classes: {np.unique(y_train_flat)}")
    print(f"Val pixels: {X_val_flat.shape[0]}, classes: {np.unique(y_val_flat)}")
    print(f"Test pixels: {X_test_flat.shape[0]}, classes: {np.unique(y_test_flat)}")

    # Save separate npz files
    np.savez("data/train_pixelwise.npz", X=X_train_flat, y=y_train_flat)
    np.savez("data/val_pixelwise.npz", X=X_val_flat, y=y_val_flat)
    np.savez("data/test_pixelwise.npz", X=X_test_flat, y=y_test_flat)

    print("Saved: train_pixelwise.npz, val_pixelwise.npz and test_pixelwise.npz")

if __name__ == "__main__":
    main()
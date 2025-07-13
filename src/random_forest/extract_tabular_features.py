import h5py
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def flatten_samples(X, Y):
    N, T, C, H, W = X.shape
    X = X.permute(0, 3, 4, 1, 2).reshape(N * H * W, T * C)
    Y = Y.reshape(N * H * W)
    return X.numpy(), Y.numpy()

def main():
    with h5py.File("data/dataset_val_train.h5", 'r') as hf:
        X_valtrain = torch.tensor(hf['data'][:])
        Y_valtrain = torch.tensor(hf['labels'][:])

    with h5py.File("data/dataset_test.h5", 'r') as hf:
        X_test = torch.tensor(hf['data'][:])
        Y_test = torch.tensor(hf['labels'][:])

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

    X_train = X_valtrain[list(train_indices)]
    Y_train = Y_valtrain[list(train_indices)]
    X_val = X_valtrain[val_indices]
    Y_val = Y_valtrain[val_indices]

    # ⚡️ Ora molto più veloce
    X_train_flat, y_train_flat = flatten_samples(X_train, Y_train)
    X_val_flat, y_val_flat = flatten_samples(X_val, Y_val)
    X_test_flat, y_test_flat = flatten_samples(X_test, Y_test)

    print(f"Train pixels: {X_train_flat.shape[0]}, classes: {np.unique(y_train_flat)}")
    print(f"Val pixels: {X_val_flat.shape[0]}, classes: {np.unique(y_val_flat)}")
    print(f"Test pixels: {X_test_flat.shape[0]}, classes: {np.unique(y_test_flat)}")

    np.savez("data/train_pixelwise.npz", X=X_train_flat, y=y_train_flat)
    np.savez("data/val_pixelwise.npz", X=X_val_flat, y=y_val_flat)
    np.savez("data/test_pixelwise.npz", X=X_test_flat, y=y_test_flat)
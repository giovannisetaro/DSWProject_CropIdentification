import h5py
import torch
import numpy as np
from collections import defaultdict
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

    N = X.shape[0]
    all_indices = list(range(N))
    
    # 2. Build a mapping from class -> list of samples that contain that class
    class_to_samples = defaultdict(set)
    for i in range(N):
        unique_classes = torch.unique(Y[i])
        for cls in unique_classes:
            class_to_samples[int(cls.item())].add(i)

    # 3. Select the smallest subset of samples that cover all classes
    train_indices = set()
    covered_classes = set()

    while len(covered_classes) < len(class_to_samples):
        best_sample = None
        best_new_classes = set()
        for i in all_indices:
            if i in train_indices:
                continue
            sample_classes = set(torch.unique(Y[i]).tolist())
            new_classes = sample_classes - covered_classes
            if len(new_classes) > len(best_new_classes):
                best_sample = i
                best_new_classes = new_classes
        if best_sample is None:
            break
        train_indices.add(best_sample)
        covered_classes.update(torch.unique(Y[best_sample]).tolist())

    # 4. Split remaining samples into validation and test sets
    remaining_indices = [i for i in all_indices if i not in train_indices]
    np.random.seed(42)
    np.random.shuffle(remaining_indices)
    half = len(remaining_indices) // 2
    val_indices = set(remaining_indices[:half])
    test_indices = set(remaining_indices[half:])

    assert len(train_indices & val_indices) == 0
    assert len(train_indices & test_indices) == 0
    assert len(val_indices & test_indices) == 0

    # 5. Extract the splits
    X_train = [X[i] for i in train_indices]
    Y_train = [Y[i] for i in train_indices]
    X_val = [X[i] for i in val_indices]
    Y_val = [Y[i] for i in val_indices]
    X_test = [X[i] for i in test_indices]
    Y_test = [Y[i] for i in test_indices]

    print(f"Train samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    # 6. Flatten and save
    X_train_flat, y_train_flat = flatten_samples(X_train, Y_train)
    X_val_flat, y_val_flat = flatten_samples(X_val, Y_val)
    X_test_flat, y_test_flat = flatten_samples(X_test, Y_test)

    print(f"Train pixels: {X_train_flat.shape[0]}, classes: {np.unique(y_train_flat)}")
    print(f"Val pixels: {X_val_flat.shape[0]}, classes: {np.unique(y_val_flat)}")
    print(f"Test pixels: {X_test_flat.shape[0]}, classes: {np.unique(y_test_flat)}")

    np.savez("data/train_pixelwise.npz", X=X_train_flat, y=y_train_flat)
    np.savez("data/val_pixelwise.npz", X=X_val_flat, y=y_val_flat)
    np.savez("data/test_pixelwise.npz", X=X_test_flat, y=y_test_flat)

if __name__ == "__main__":
    main()

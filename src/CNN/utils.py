import torch
import numpy as np
from collections import Counter

def compute_class_weights(dataset, num_classes, ignore_index=0):
    """
    Compute normalized class weights (excluding the ignore_index class).
    This helps balance the loss when training on imbalanced datasets.

    Args:
        dataset (torch.utils.data.Dataset): Dataset returning (x, y) where y is a label mask.
        num_classes (int): Total number of classes including background.
        ignore_index (int): Label to ignore (typically 0 for background).

    Returns:
        torch.FloatTensor: Normalized class weights of shape [num_classes].
    """
    class_counts = torch.zeros(num_classes, dtype=torch.long)

    for _, label in dataset:
        labels_flat = label.flatten()
        labels_flat = labels_flat[labels_flat != ignore_index]  # exclude background
        counts = torch.bincount(labels_flat, minlength=num_classes)
        class_counts += counts

    # Avoid division by zero
    class_counts[class_counts == 0] = 1

    weights = 1.0 / class_counts.float()
    weights[ignore_index] = 0.0  # set weight for ignored class to zero

    # Normalize weights to sum to 1 (optional, but helps for stability)
    weights = weights / weights.sum()

    return weights

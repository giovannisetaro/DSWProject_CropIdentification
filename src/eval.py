import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
import seaborn as sns

def plot_confusion_matrix(cm, class_names=None, figsize=(8, 6)):
    """
    Plots the confusion matrix `cm` (torch.Tensor or ndarray).
    class_names: list of class names (optional).
    """
    # Convert to numpy if necessary
    if isinstance(cm, torch.Tensor):
        cm = cm.cpu().numpy()

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


def compute_metrics_from_CM(confusion_matrix, total_loss=0.0, data_length=None):
    """
    From a confusion_matrix torch.Tensor [C, C], computes and displays:
      - average loss (if total_loss and data_length are provided)
      - global accuracy
      - precision, recall, macro F1-score
    Returns a dictionary of metrics.
    """
    # Ensure compatibility: convert to long Tensor if needed
    if not isinstance(confusion_matrix, torch.Tensor):
        confusion_matrix = torch.tensor(confusion_matrix, dtype=torch.long)
    cm = confusion_matrix

    # Totals
    total_pixels = cm.sum().item()
    if data_length is None:
        data_length = total_pixels

    # Accuracy
    correct = cm.diag().sum().item()
    accuracy = correct / total_pixels

    # True positives, false positives, false negatives
    TP = cm.diag().float()
    FP = cm.sum(dim=0).float() - TP
    FN = cm.sum(dim=1).float() - TP
    eps = 1e-12

    precision = TP / (TP + FP + eps)
    recall    = TP / (TP + FN + eps)
    f1        = 2 * (precision * recall) / (precision + recall + eps)

    # Macro averages
    precision_macro = precision.mean().item()
    recall_macro    = recall.mean().item()
    f1_macro        = f1.mean().item()

    # Average loss if requested
    avg_loss = (total_loss / data_length) if data_length > 0 else None

    # Display
    if avg_loss is not None:
        print(f"Average Loss       : {avg_loss:.4f}")
    print(f"Global Accuracy    : {accuracy:.4f}")
    print(f"Macro Precision    : {precision_macro:.4f}")
    print(f"Macro Recall       : {recall_macro:.4f}")
    print(f"Macro F1-score     : {f1_macro:.4f}")

    return {
        'loss'            : avg_loss,
        'accuracy'        : accuracy,
        'precision_macro' : precision_macro,
        'recall_macro'    : recall_macro,
        'f1_macro'        : f1_macro,
    }


def evaluate(y_true, y_pred, num_classes=None, class_names=None,
             total_loss=0.0, data_length=None, plot_cm=True):
    """
    Generic evaluation function: takes y_true and y_pred (list, np.ndarray, or torch.Tensor),
    reconstructs the confusion matrix, then computes and displays metrics.

    Arguments:
      - y_true, y_pred : flat integer vectors [0..C-1]
      - num_classes    : number of classes (if None, determined automatically)
      - class_names    : class names for CM display
      - total_loss     : (optional) total loss if computed
      - data_length    : (optional) dataset size for average loss
      - plot_cm        : True to display the confusion matrix

    Returns:
      - cm       : torch.Tensor [C, C]
      - metrics  : dictionary of metrics (see compute_metrics_from_CM)
    """
    # Convert to numpy
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()

    # Determine num_classes
    if num_classes is None:
        num_classes = max(y_true.max(), y_pred.max()) + 1

    # 1) Build confusion matrix using sklearn
    cm_np = sk_confusion_matrix(
        y_true, y_pred,
        labels=list(range(num_classes))
    )

    # 2) Convert to torch.Tensor
    import torch
    cm = torch.tensor(cm_np, dtype=torch.long)

    # 3) Plot if requested
    if plot_cm:
        plot_confusion_matrix(cm, class_names=class_names)

    # 4) Compute metrics
    metrics = compute_metrics_from_CM(cm, total_loss=total_loss, data_length=data_length)

    return cm, metrics
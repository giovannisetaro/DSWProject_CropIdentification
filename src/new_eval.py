import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data import get_dataset_splits_from_h5
from xgboost import XGBClassifier
import joblib  # Pour charger un modèle XGBoost si besoin

# CHATGPT OUTPUT AS EXAMPLE


def plot_confusion_matrix(cm):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm.numpy(), annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix (pixel-wise)")
    plt.tight_layout()
    plt.show()


def evaluate_xgb_predictions(y_preds, dataloader, num_classes):
    # Initialize confusion matrix
    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    total_pixels = 0
    all_preds = []
    all_labels = []

    idx = 0  # pour itérer dans les prédictions plates

    for _, y in dataloader:
        batch_size, H, W = y.shape
        n_pixels = batch_size * H * W

        # Flatten labels
        y_flat = y.view(-1)

        # Slice the corresponding predicted labels (déjà plats)
        preds_flat = torch.tensor(y_preds[idx:idx + n_pixels])
        idx += n_pixels

        all_preds.append(preds_flat)
        all_labels.append(y_flat)

        for true, pred in zip(y_flat, preds_flat):
            confusion_matrix[true.long(), pred.long()] += 1

        total_pixels += y_flat.numel()

    # Compute metrics
    correct = confusion_matrix.diag().sum().item()
    accuracy = correct / total_pixels

    TP = confusion_matrix.diag().float()
    FP = confusion_matrix.sum(dim=0).float() - TP
    FN = confusion_matrix.sum(dim=1).float() - TP

    epsilon = 1e-12
    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    precision_macro = precision.mean().item()
    recall_macro = recall.mean().item()
    f1_macro = f1.mean().item()

    # Display results
    print(f"Pixel Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {precision_macro:.4f}")
    print(f"Macro Recall: {recall_macro:.4f}")
    print(f"Macro F1-score: {f1_macro:.4f}")

    plot_confusion_matrix(confusion_matrix)

    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
    }


def main():
    # === Load tabular test data to get features ===
    from data import get_dataset_3splits  # Pour extraire X_test
    _, _, tabular_loader = get_dataset_3splits(
        h5_path='data/Dataset.h5',
        dataset_type="rf",
        test_split_on="zone",
        batch_size=4096  # batch large pour tout charger
    )

    # === Extraire X_test (features) ===
    X_test = []
    for X_batch, _ in tabular_loader:
        X_test.append(X_batch.numpy())
    X_test = np.concatenate(X_test, axis=0)

    # === Charger modèle XGBoost ===
    # Option 1 : depuis fichier .json
    xgb_model = XGBClassifier()
    xgb_model.load_model("xgb_model.json")

    # Option 2 : depuis pickle
    # xgb_model = joblib.load("xgb_model.pkl")

    # === Prédictions sur tous les pixels du test ===
    y_preds = xgb_model.predict(X_test)  # y_preds plat, shape [num_pixels]

    # === Recharger les labels CNN pour comparer ===
    # Important : ce dataloader doit avoir les mêmes indices test que celui du `rf`
    _, test_loader = get_dataset_splits_from_h5('data/Dataset.h5', batch_size=8)

    # === Evaluer ===
    num_classes = len(np.unique(y_preds))  # ou fixe : 26
    evaluate_xgb_predictions(y_preds, test_loader, num_classes)


if __name__ == "__main__":
    main()

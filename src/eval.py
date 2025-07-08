import torch
import torch.nn as nn
from CNN_Model import CropTypeClassifier  
from data import get_dataset_splits_from_h5
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_confusion_matrix(cm):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm.numpy(), annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix (pixel-wise)")
    plt.tight_layout()
    plt.show()


def evaluate(model, dataloader, device, num_classes):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    # Confusion matrix (rows = true classes, cols = predicted classes)
    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    total_loss = 0.0
    total_pixels = 0

    with torch.no_grad():
        # Loop on every single batch (batch size = 8)
        for x, y in dataloader:
            # x = input [Batch size, Bands, temporal step, H, W]
            # y = real classes [Batch size, H, W] 
            x, y = x.to(device), y.to(device)

            # Retrive the logit [Batch size, number of classes, H, W]
            outputs = model(x)

            loss = criterion(outputs, y)
            
            # Compute total_loss by multiplying loss by batch size,
            # will be useful to compute the average later on
            total_loss += loss.item() * x.size(0)

            # Prediction is the highest probability class
            preds = outputs.argmax(dim=1)

            # Flatten for confusion matrix to enable comparison
            preds_flat = preds.view(-1)
            labels_flat = y.view(-1)

            # Update confusion matrix
            for true, predict in zip(labels_flat, preds_flat):
                confusion_matrix[true.long(), predict.long()] += 1

            total_pixels += labels_flat.numel()

    # Compute global accuracy
    correct = confusion_matrix.diag().sum().item()
    accuracy = correct / total_pixels

    # Compute per-class precision, recall, F1
    # Here, TP, FP and FN are list
    TP = confusion_matrix.diag().float()

    # Since rows = true classes and cols = predicted classes,
    # sum it and remove TP cell of the confusion matrix
    FP = confusion_matrix.sum(dim=0).float() - TP
    FN = confusion_matrix.sum(dim=1).float() - TP

    epsilon = 1e-12 # Easiest way to avoid 0 div
    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    # Macro-averaged metrics
    precision_macro = precision.mean().item()
    recall_macro = recall.mean().item()
    f1_macro = f1.mean().item()
    avg_loss = total_loss / len(dataloader.dataset)

    # Display metrics
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Validation Pixel Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {precision_macro:.4f}")
    print(f"Macro Recall: {recall_macro:.4f}")
    print(f"Macro F1-score: {f1_macro:.4f}")

    plot_confusion_matrix(confusion_matrix)

    # Return metrics in a dict, may be uses later on
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Charger le dataset validation (adapte selon ta fonction)
    _, test_loader = get_dataset_splits_from_h5('data/Dataset.h5', batch_size=8)
    
    model = CropTypeClassifier(num_classes=26)
    model.load_state_dict(torch.load('checkpoints/crop_model_epoch1.pth'))
    model.to(device)

    evaluate(model, val_loader, device, num_classes)


if __name__ == "__main__":
    main()

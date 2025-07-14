import torch
import torch.nn as nn
from torch.optim import Adam
from src.data import get_dataset_3splits
from src.CNN.CNN_Model import CropTypeClassifier
from torch.utils.data import DataLoader, Subset, ConcatDataset
from sklearn.model_selection import KFold
import os
from tqdm import tqdm
from src.eval import evaluate
import itertools
from src.CNN.utils import compute_class_weights

def train_loop(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x, y in tqdm(dataloader, desc="Training", leave=False):
        # Move inputs and labels to device with non-blocking to speed up transfer
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_model_on_loader(model, dataloader, device, criterion, num_classes=26, class_names=None, plot_cm=False):
    model.eval()
    y_true = []
    y_pred = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)
            preds = outputs.argmax(dim=1)
            y_true.append(y.view(-1).cpu())
            y_pred.append(preds.view(-1).cpu())

    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    _, metrics = evaluate(
        y_true=y_true,
        y_pred=y_pred,
        num_classes=num_classes,
        class_names=class_names,
        total_loss=total_loss,
        data_length=total_samples,
        plot_cm=plot_cm
    )
    return metrics

def main():
    # Select device: GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Number of classes to predict (updated from 26 to 50)
    num_classes = 50

    # Load training, validation, and test datasets from .h5 files
    train_dataset, val_dataset, test_dataset = get_dataset_3splits(
        "data/dataset_val_train.h5",
        "data/dataset_test.h5",
        val_ratio=0.15,     # 15% of training data reserved for validation
        debug=False,         # Enable debug mode for additional checks/logs
        num_classes=num_classes  # Pass the number of classes explicitly
    )

    # Optional debugging: check label range on a sample of training data
    if True:
        all_labels = []
        for i in range(min(100, len(train_dataset))):  # check only first 100 samples for speed
            _, y = train_dataset[i]
            all_labels.append(y)
        # Flatten and concatenate all label tensors to check global min/max
        all_labels = torch.cat([y.flatten() for y in all_labels])
        print(f"[DEBUG] Train labels: min={all_labels.min().item()}, max={all_labels.max().item()}")
        # Assert that all labels are within valid range [0, num_classes-1]
        assert all_labels.min() >= 0 and all_labels.max() < num_classes, "Labels out of range!"

    # Combine train and val datasets for K-Fold cross-validation splitting
    train_val_dataset = ConcatDataset([train_dataset, val_dataset])

    # Setup 5-fold cross-validation with shuffle and fixed random seed for reproducibility
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Hyperparameter grid for learning rate and kernel size
    learning_rates = [1e-3, 5e-4, 1e-4]
    kernel_sizes = [3, 5, 7]

    # Training parameters
    batch_size = 128
    num_epochs = 50
    patience = 3          # Early stopping patience (number of epochs to wait for improvement)
    min_delta = 1e-4      # Minimum decrease in validation loss to qualify as improvement

    # Hyperparameter optimization loop: iterate over all lr and kernel size combinations
    for lr, kernel_size in itertools.product(learning_rates, kernel_sizes):
        print(f"\n[HPO] lr={lr}, kernel_size={kernel_size}")

        # Perform K-Fold cross-validation splits
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_dataset)):
            print(f"\nFold {fold + 1} | lr={lr}, kernel_size={kernel_size}")

            # Create subset datasets for current fold's train and validation indices
            train_subset = Subset(train_val_dataset, train_idx)
            val_subset = Subset(train_val_dataset, val_idx)

            # Create data loaders for train and validation sets
            train_loader = DataLoader(
                train_subset, batch_size=batch_size, shuffle=True,
                num_workers=8, pin_memory=True
            )
            val_loader = DataLoader(
                val_subset, batch_size=batch_size, shuffle=False,
                num_workers=8, pin_memory=True
            )

            # Initialize model with given number of classes and kernel size; move to device
            model = CropTypeClassifier(num_classes=num_classes, kernel_size=kernel_size).to(device)

            # Define loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = Adam(model.parameters(), lr=lr)

            # Initialize variables for early stopping tracking
            best_val_loss = float('inf')
            best_epoch = 0
            early_stop_counter = 0
            best_model_path = None

            # Training loop for each epoch
            for epoch in range(1, num_epochs + 1):
                # Train for one epoch and get training loss
                train_loss = train_loop(model, train_loader, optimizer, criterion, device)

                # Evaluate on validation set: get metrics including loss and accuracy
                metrics = evaluate_model_on_loader(model, val_loader, device, criterion, num_classes=num_classes)
                val_loss = metrics['loss']
                val_acc = metrics['accuracy']

                # Print current epoch stats
                print(f"[Fold {fold+1} | Epoch {epoch}/{num_epochs}] Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

                # Check if validation loss improved significantly (by at least min_delta)
                if val_loss + min_delta < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    early_stop_counter = 0  # reset patience counter

                    # Prepare directory and save current best model
                    subdir = f"checkpoints/lr_{lr}_ks_{kernel_size}"
                    os.makedirs(subdir, exist_ok=True)
                    best_model_path = os.path.join(subdir, f"fold_{fold+1}.pth")
                    torch.save(model.state_dict(), best_model_path)
                else:
                    # No improvement, increment early stopping counter
                    early_stop_counter += 1
                    print(f"No improvement. Early stop counter: {early_stop_counter}/{patience}")

                # Stop training if no improvement for 'patience' epochs
                if early_stop_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            # After training all epochs or early stop, print best epoch info and model path
            print(f"Best model for Fold {fold+1}: epoch {best_epoch}, val loss: {best_val_loss:.4f}")
            if best_model_path:
                print(f"Model saved at: {best_model_path}")


if __name__ == "__main__":
    main()

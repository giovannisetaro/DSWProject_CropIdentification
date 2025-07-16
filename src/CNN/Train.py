import torch
import torch.nn as nn
from torch.optim import Adam
from src.CNN.CNN_Model import CropTypeClassifier
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import os
from tqdm import tqdm
from src.eval import evaluate
import itertools
from torch.utils.data import ConcatDataset
from src.data import IndexedDataset
from torch.utils.data import Dataset, TensorDataset

def train_loop(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for x, y in tqdm(dataloader, desc="Training", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate_model_on_loader(model, dataloader, device, criterion, num_classes=51, class_names=None, plot_cm=False):
    model.eval()
    y_true = []
    y_pred = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    import h5py

    with h5py.File("data/dataset_val_train.h5", "r") as f:
        X = torch.tensor(f["data"][:])
        Y = torch.tensor(f["labels"][:]).long()
        print(f"X shape: {X.shape}")
        print(f"Y shape: {Y.shape}")
    train_val_dataset = TensorDataset(X, Y)


    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    learning_rates = [1e-3, 1e-4]
    kernel_sizes = [3, 5]  
    batch_size = 64
    num_epochs = 50  
    patience = 3
    min_delta = 1e-4


    best_overall_val_loss = float('inf')
    best_overall_model_path = None
    best_overall_hparams = None
    best_overall_fold = None
    best_overall_epoch = None

    for lr, kernel_size in itertools.product(learning_rates, kernel_sizes):

        print(f"\n[HPO] lr={lr}, kernel_size={kernel_size}")

        for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_dataset)):
            print(f"\nFold {fold + 1} | lr={lr}, kernel_size={kernel_size}, patience={patience}")

            train_dataset_fold = IndexedDataset(train_val_dataset, train_idx)
            val_dataset_fold = IndexedDataset(train_val_dataset, val_idx)

            train_loader = DataLoader(
                train_dataset_fold,
                batch_size=batch_size,
                shuffle=True,
                num_workers=8,
                pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset_fold,
                batch_size=batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True
            )

            model = CropTypeClassifier(num_classes=51, kernel_size=kernel_size).to(device)
            criterion = nn.CrossEntropyLoss(ignore_index=255)
            optimizer = Adam(model.parameters(), lr=lr)

            best_val_loss = float('inf')
            best_epoch = 0
            early_stop_counter = 0
            best_model_state = None  # Store the best model state dict for this fold

            for epoch in range(1, num_epochs + 1):
                train_loss = train_loop(model, train_loader, optimizer, criterion, device)
                metrics = evaluate_model_on_loader(model, val_loader, device, criterion, num_classes=51)
                val_loss = metrics['loss']
                val_acc = metrics['accuracy']

                print(f"[Fold {fold+1} | Epoch {epoch}/{num_epochs}] "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

                # Check if validation loss improved beyond min_delta
                if val_loss + min_delta < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    best_model_state = model.state_dict()  # Save model state in memory
                    early_stop_counter = 0

                    # Update global best model if current validation loss is lower
                    if val_loss < best_overall_val_loss:
                        best_overall_val_loss = val_loss
                        best_overall_model_path = "models/best_model_cnn.pth"
                        best_overall_hparams = (lr, kernel_size)
                        best_overall_fold = fold + 1
                        best_overall_epoch = epoch
                        os.makedirs("models", exist_ok=True)
                        torch.save(model.state_dict(), best_overall_model_path)  # Save global best model immediately
                else:
                    early_stop_counter += 1
                    print(f"No improvement. Early stop counter: {early_stop_counter}/{patience}")

                if early_stop_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            # Save the best model for this fold only once after training completes
            if best_model_state is not None:
                subdir = f"checkpoints/lr_{lr}_ks_{kernel_size}"
                os.makedirs(subdir, exist_ok=True)
                best_model_path = os.path.join(subdir, f"fold_{fold+1}.pth")
                torch.save(best_model_state, best_model_path)
                print(f"Saved best fold model to {best_model_path} at epoch {best_epoch}")

            print(f"\nBest model for Fold {fold+1}: epoch {best_epoch}, val loss: {best_val_loss:.4f}")


    print("\n=== Best overall model summary ===")
    print(f"Best overall model saved at: {best_overall_model_path}")
    print(f"Parameters: lr={best_overall_hparams[0]}, kernel_size={best_overall_hparams[1]}, fold={best_overall_fold}, epoch={best_overall_epoch}")
    print(f"Validation loss: {best_overall_val_loss:.4f}")

if __name__ == "__main__":
    main()

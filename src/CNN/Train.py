import os
import itertools
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset, ConcatDataset
from sklearn.model_selection import KFold
from tqdm import tqdm

from src.data import get_dataset_3splits, CropCnnDataset
from src.CNN.CNN_Model import CropTypeClassifier
from src.eval import evaluate

IGNORE_INDEX = 255
NUM_CLASSES = 51

# HPO grid (no batch size)
LEARNING_RATES = [1e-3, 5e-4]
KERNEL_SIZES = [3, 5]
BATCH_SIZE = 30  # fixed

# Fixed training settings
NUM_EPOCHS = 50
NUM_WORKERS = 4
PATIENCE = 5
MIN_DELTA = 1e-4

def train_loop(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_samples = 0

    for x, y in tqdm(dataloader, desc="Training", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    return avg_loss

def evaluate_model_on_loader(model, dataloader, device, criterion, num_classes=NUM_CLASSES, class_names=None, plot_cm=False):
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
    torch.backends.cudnn.benchmark = True

    # Load training/validation dataset
    split_files = [
        f"data/split_datasets/dataset_split_{i}.h5"
        for i in range(8)
    ]
    datasets = [CropCnnDataset(fp) for fp in split_files]
    train_val_dataset = ConcatDataset(datasets)

    # K-Fold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Grid search over learning rate and kernel size
    for lr, kernel_size in itertools.product(LEARNING_RATES, KERNEL_SIZES):
        print(f"\n HPO Trial: LR={lr}, Kernel={kernel_size}, Batch={BATCH_SIZE}")

        for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_dataset)):
            print(f"\n Fold {fold + 1}")

            train_subset = Subset(train_val_dataset, train_idx)
            val_subset = Subset(train_val_dataset, val_idx)

            train_loader = DataLoader(
                train_subset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=NUM_WORKERS,
                pin_memory=True,
                drop_last=False
            )
            val_loader = DataLoader(
                val_subset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=True,
                drop_last=False
            )

            model = CropTypeClassifier(num_classes=NUM_CLASSES, kernel_size=kernel_size).to(device)

            assert next(model.parameters()).is_cuda, (
                " Model is NOT on CUDA! Check model.to(device)"
            )

            criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
            optimizer = Adam(model.parameters(), lr=lr)

            best_val_loss = float('inf')
            best_epoch = 0
            early_stop_counter = 0
            best_model_path = None

            for epoch in range(1, NUM_EPOCHS + 1):
                train_loss = train_loop(model, train_loader, optimizer, criterion, device)
                metrics = evaluate_model_on_loader(model, val_loader, device, criterion, num_classes=NUM_CLASSES)
                val_loss = metrics['loss']
                val_acc = metrics.get('accuracy', 0)

                print(f"[HPO: lr={lr}, ks={kernel_size} | Fold {fold+1} | Epoch {epoch}/{NUM_EPOCHS}] "
                      f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

                if val_loss + MIN_DELTA < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    early_stop_counter = 0
                    subdir = f"checkpoints/lr_{lr}_ks_{kernel_size}_bs_{BATCH_SIZE}"
                    os.makedirs(subdir, exist_ok=True)
                    best_model_path = os.path.join(subdir, f"fold_{fold+1}.pth")
                    torch.save(model.state_dict(), best_model_path)
                else:
                    early_stop_counter += 1
                    print(f"No improvement. Early stop counter: {early_stop_counter}/{PATIENCE}")

                if early_stop_counter >= PATIENCE:
                    print(f" Early stopping at epoch {epoch}")
                    break

            print(f"\n Best model for Fold {fold+1}: Epoch {best_epoch}, Val Loss: {best_val_loss:.4f}")
            if best_model_path:
                print(f"Model saved at: {best_model_path}")

if __name__ == "__main__":
    main()

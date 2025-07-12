import torch
import torch.nn as nn
from torch.optim import Adam
from src.data import get_dataset_3splits
from src.CNN.CNN_Model import CropTypeClassifier
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import os
from tqdm import tqdm
from src.eval import evaluate
import itertools
from torch.utils.data import ConcatDataset

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


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset, val_dataset, _ = get_dataset_3splits("data/Dataset.h5", val_ratio=0.15, test_ratio=0.15)
    train_val_dataset = ConcatDataset([train_dataset, val_dataset])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    learning_rates = [1e-3, 5e-4]
    batch_sizes = [8, 16]
    num_epochs = 10
    patience = 3
    min_delta = 1e-4

    for lr, batch_size in itertools.product(learning_rates, batch_sizes):
        print(f"\n[HPO] lr={lr}, batch_size={batch_size}")

        for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_dataset)):
            print(f"\nFold {fold + 1} | lr={lr}, bs={batch_size}, epochs={num_epochs}, patience={patience}")

            train_subset = Subset(train_val_dataset, train_idx)
            val_subset = Subset(train_val_dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

            model = CropTypeClassifier(num_classes=26).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = Adam(model.parameters(), lr=lr)

            best_val_loss = float('inf')
            best_epoch = 0
            early_stop_counter = 0
            best_model_path = None

            for epoch in range(1, num_epochs + 1):
                train_loss = train_loop(model, train_loader, optimizer, criterion, device)
                metrics = evaluate(model, val_loader, device=device, num_classes=26)
                val_loss = metrics['loss']
                val_acc = metrics['accuracy']

                print(f"[Fold {fold+1} | Epoch {epoch}/{num_epochs}] "
                      f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

                if val_loss + min_delta < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    early_stop_counter = 0
                    subdir = f"checkpoints/lr_{lr}_bs_{batch_size}"
                    os.makedirs(subdir, exist_ok=True)
                    best_model_path = os.path.join(subdir, f"fold_{fold+1}.pth")
                    torch.save(model.state_dict(), best_model_path)
                else:
                    early_stop_counter += 1
                    print(f"No improvement. Early stop counter: {early_stop_counter}/{patience}")

                if early_stop_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            print(f"\nBest model for Fold {fold+1}: epoch {best_epoch}, val loss: {best_val_loss:.4f}")
            if best_model_path:
                print(f"Model saved at: {best_model_path}")


if __name__ == "__main__":
    main()

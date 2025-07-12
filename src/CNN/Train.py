import torch
import torch.nn as nn
from torch.optim import Adam
from data import get_dataset_3splits
from CNN.CNN_Model import CropTypeClassifier
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import os
from tqdm import tqdm
from eval import evaluate
import itertools

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


    # Load the dataset and split off the test set
    train_val_dataset, _ = get_dataset_3splits("data/Dataset.h5",val_ratio = 0.15, test_ratio=0.15) 

    # Set up K-Fold Cross Validation (5 folds)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)  

    # Hyperparameter search space
    learning_rates = [1e-3, 5e-4]
    batch_sizes = [8, 16]
    epoch_options = [5, 10, 15]
    early_stopping_patience_list = [3, 5, 7, 10]
    min_delta = 1e-4  # Minimum change to qualify as an improvement

    for lr, batch_size, num_epochs, patience in itertools.product(
            learning_rates, batch_sizes, epoch_options, early_stopping_patience_list):
        
        print(f"\n[HPO] lr={lr}, batch_size={batch_size}, num_epochs={num_epochs}, patience={patience}")

        for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_dataset)):
            print(f"Fold {fold + 1} | lr={lr}, bs={batch_size}, epochs={num_epochs}, patience={patience}")

            train_subset = Subset(train_val_dataset, train_idx)
            val_subset = Subset(train_val_dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

            model = CropTypeClassifier(num_classes=26).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = Adam(model.parameters(), lr=lr)

            os.makedirs("checkpoints", exist_ok=True)

            best_val_loss = float('inf')
            best_epoch = 0
            early_stop_counter = 0
            best_model_path = None

            for epoch in range(1, num_epochs + 1):
                train_loss = train_loop(model, train_loader, optimizer, criterion, device)
                val_loss, val_acc = evaluate(model, val_loader, criterion, device)

                print(f"[Fold {fold+1} | Epoch {epoch}/{num_epochs}] "
                      f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

                # Early stopping logic
                if val_loss + min_delta < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    early_stop_counter = 0
                    best_model_path = (
                        f"checkpoints/best_model_fold{fold+1}_lr{lr}_bs{batch_size}_ne{num_epochs}_pat{patience}.pth"
                    )
                    torch.save(model.state_dict(), best_model_path)
                else:
                    early_stop_counter += 1
                    print(f"No improvement. Early stop counter: {early_stop_counter}/{patience}")

                if early_stop_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            print(f"Best model for Fold {fold+1}: epoch {best_epoch}, val loss: {best_val_loss:.4f}")
            if best_model_path:
                print(f"Model saved at: {best_model_path}")

if __name__ == "__main__":
    main()
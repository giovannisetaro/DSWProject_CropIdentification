import torch
import torch.nn as nn
from torch.optim import Adam
from data import get_dataset_splits_from_h5
from CNN.CNN_Model import CropTypeClassifier
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import os
from tqdm import tqdm

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
    train_val_dataset, _ = get_dataset_splits_from_h5("data/Dataset.h5", test_ratio=0.1) 

    # Set up K-Fold Cross Validation (5 folds)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)   

    # Iterate over each fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_dataset)):
        print(f"\nFold {fold + 1}")

        # Create train and validation subsets
        train_subset = Subset(train_val_dataset, train_idx)
        val_subset = Subset(train_val_dataset, val_idx)

        # Create DataLoaders
        train_loader = DataLoader(train_subset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=8, shuffle=False)

        # Initialize model, loss function and optimizer
        model = CropTypeClassifier(num_classes=26).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=1e-3)

        # Create checkpoints directory if it doesn't exist
        os.makedirs("checkpoints", exist_ok=True)

        # Training loop (can increase number of epochs)
        for epoch in range(1, 2):
            train_loss = train_loop(model, train_loader, optimizer, criterion, device)
            print(f"Epoch {epoch}, Fold {fold+1}, Train loss: {train_loss:.4f}")

            # Save model weights after each fold and epoch
            checkpoint_path = f"checkpoints/crop_model_fold{fold+1}_epoch{epoch}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model saved to {checkpoint_path}")

if __name__ == "__main__":
    main()
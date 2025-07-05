import torch
import torch.nn as nn
from torch.optim import Adam
from data import get_dataloaders_from_h5
from CNN_Model import CropTypeClassifier
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
    train_loader, val_loader = get_dataloaders_from_h5('data/Dataset.h5', batch_size=8)      #val_loader :Il faut calculer la perte (et métriques) sur la validation après chaque epoch pour monitorer le surapprentissage.
    model = CropTypeClassifier(num_classes=26).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)

    os.makedirs("checkpoints", exist_ok=True)  # crée dossier checkpoints si absent

    for epoch in range(1, 2):
        train_loss = train_loop(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch}, Train loss: {train_loss:.4f}")

        # Sauvegarde les poids après chaque epoch
        checkpoint_path = f"checkpoints/crop_model_epoch{epoch}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")

if __name__ == "__main__":
    main()
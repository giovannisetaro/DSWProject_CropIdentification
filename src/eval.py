import torch
import torch.nn as nn
from CNN_Model import CropTypeClassifier  # adapte selon ton projet
from data import get_dataloaders_from_h5

def evaluate(model, dataloader, device):
    model.eval()
    total_correct = 0
    total_pixels = 0
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item() * x.size(0)
            
            preds = outputs.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_pixels += y.numel()
    
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = total_correct / total_pixels
    
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Validation Pixel Accuracy: {accuracy:.4f}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Charger le dataset validation (adapte selon ta fonction)
    _, val_loader = get_dataloaders_from_h5('data/Dataset.h5', batch_size=8)
    
    model = CropTypeClassifier(num_classes=26)
    model.load_state_dict(torch.load('checkpoints/crop_model_epoch1.pth'))
    model.to(device)
    
    evaluate(model, val_loader, device)

if __name__ == "__main__":
    main()
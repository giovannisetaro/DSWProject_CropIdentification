import torch
import matplotlib.pyplot as plt
from data import CropDataset  
import numpy as np
import rasterio
from CNN_Model import CropTypeClassifier
from shapely.geometry import Point, Polygon


def load_model(model_path, device):
    # Instancier ton modèle (adapte le constructeur à ta classe)
    model = CropTypeClassifier(num_classes=26)  
    # Charger les poids
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # Envoyer le modèle sur le device (CPU ou GPU)
    model.to(device)
    
    return model



def create_spatial_mosaic_from_geo_coords(patches, geo_coords, patch_size, transform):
    pixel_coords = [rasterio.transform.rowcol(transform, lon, lat) for lon, lat in geo_coords]
    rows = [r for r, c in pixel_coords]
    cols = [c for r, c in pixel_coords]

    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)

    height = max_row - min_row + patch_size[0]
    width = max_col - min_col + patch_size[1]

    # Si multi-canal, prévoir un tableau 3D (C, H, W), ici pour 2D exemple simple:
    mosaic = np.zeros((height, width), dtype=patches[0].dtype)

    for patch, (row, col) in zip(patches, pixel_coords):
        row_offset = row - min_row
        col_offset = col - min_col
        mosaic[row_offset:row_offset+patch_size[0], col_offset:col_offset+patch_size[1]] = patch

    return mosaic





def select_patches_by_coords(dataset, polygon_coords):

    indices = []
    for idx in range(10):
        indices.append(idx)
    return indices



def visualize_predictions(dataset, model, selected_indices, device):
    model.eval()
    patch_size = dataset.Y.shape[1:]  # (H, W)
    preds_patches = []
    coords_patches = []

    with torch.no_grad():
        for idx in selected_indices:
            x, y = dataset[idx]
            coord = dataset.get_coord(idx)
            x = x.unsqueeze(0).to(device)  # batch=1
            output = model(x)  # shape [1, num_classes, H, W]
            pred = output.argmax(dim=1).squeeze(0).cpu().numpy()  # [H, W]
            preds_patches.append(pred)
            coords_patches.append(coord)

    # Charger la transformée affine à partir d'un raster de référence (à adapter)
    with rasterio.open('/Users/placiermoise/Documents/dsw_proj/remote sensing crop classification/data/Tif/zone1/labels_raster_masked.tif') as src:
        transform = src.transform

    mosaic = create_spatial_mosaic_from_geo_coords(preds_patches, coords_patches, patch_size, transform)

    plt.figure(figsize=(10, 10))
    plt.imshow(mosaic, cmap='tab20')  # colormap adaptée pour classes catégorielles
    plt.title('Mosaïque des prédictions dans la zone sélectionnée')
    plt.colorbar()
    plt.axis('off')
    plt.show()









if __name__ == "__main__":
    h5_path = 'data/Dataset.h5'
    model_path = 'checkpoints/crop_model_epoch1.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = CropDataset(h5_path)
    model = load_model(model_path, device)

    bbox_coords = [
    [-1.757813, 42.098222],
   [-1.757813, 46.164614],
   [5.405273, 46.164614],
   [5.405273, 42.098222],
   [-1.757813, 42.098222]]


    selected_indices = select_patches_by_coords(dataset, bbox_coords) 
    print(f"Nombre de patches sélectionnés: {len(selected_indices)}")

    visualize_predictions(dataset, model, selected_indices, device)

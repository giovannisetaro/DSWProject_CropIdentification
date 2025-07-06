import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import rasterio
import imageio
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')



    # Charger la transformée affine à partir d'un raster de référence (à adapter)
with rasterio.open('/Users/placiermoise/Documents/dsw_proj/remote sensing crop classification/data/Tif/zone1/labels_raster_masked.tif') as src:
    transform_example = src.transform

def create_spatial_mosaic_from_geo_coords(patches, geo_coords, patch_size, transform):
    pixel_coords = [rasterio.transform.rowcol(transform, lon, lat) for lon, lat in geo_coords]
    rows = [r for r, c in pixel_coords]
    cols = [c for r, c in pixel_coords]

    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)

    height = max_row - min_row + patch_size[0]
    width = max_col - min_col + patch_size[1]

    mosaic = np.full((height, width), -1, dtype=patches[0].dtype)

    for patch, (row, col) in zip(patches, pixel_coords):
        row_offset = row - min_row
        col_offset = col - min_col
        mosaic[row_offset:row_offset+patch_size[0], col_offset:col_offset+patch_size[1]] = patch

    return mosaic


def generate_temporal_gif_from_h5(h5_path, zone_name, band_index, patch_size, out_name, vmin=None, vmax=None):
    """
    Crée un GIF temporel pour une bande et une zone données à partir d’un fichier .h5.
    
    Params:
        h5_path (str): Chemin du fichier HDF5
        zone_name (str): Nom de la zone (e.g., 'zone3')
        band_index (int): Index de la bande (e.g., 2 pour band 2)
        patch_size (tuple): Taille des patchs (H, W)
        gif_path (str): Chemin de sortie du GIF
        vmin, vmax (int/float): Pour normaliser l’affichage (optionnel)
    """
    with h5py.File(h5_path, 'r') as f:
        data = f['data'][:]  # (N, T, C, H, W)
        coords = f['coords'][:]  # (N, 2)
        zones = f['zones'][:].astype(str)  # 1D array
        dates = f['dates'][:].astype(str)  # 1D array

    zone_mask = (zones == zone_name)
    if not np.any(zone_mask):
        raise ValueError(f"Aucune donnée trouvée pour la zone {zone_name}")

    zone_data = data[zone_mask]         # (N, T, C, H, W)
    zone_coords = coords[zone_mask]     # (N, 2)
    zone_dates = dates[zone_mask]       # (N,)
    time_steps = zone_data.shape[1]

    print(f"Génération du GIF pour {zone_name} avec {time_steps} dates...")
    with rasterio.open('/Users/placiermoise/Documents/dsw_proj/remote sensing crop classification/data/Tif/zone1/labels_raster_masked.tif') as src:
        transform_example = src.transform
    # Créer les mosaïques temporelles pour la bande sélectionnée
    n=0
    for t in tqdm(range(time_steps), desc="Création des mosaïques temporelles"):
        band_patches = zone_data[:, t, band_index]  # (N, H, W)
        print("nombre de patches : ",len(band_patches))
        band_mosaic = create_spatial_mosaic_from_geo_coords(
            band_patches,
            zone_coords,
            patch_size,
            transform= transform_example # valeur fictive
        )

        plt.imshow(band_mosaic, cmap='viridis')
        plt.title(f"{zone_name}_2023_{t}")
        plt.axis('off')
        # Enregistrement du plot
        plt.savefig(f"plots/png/{out_name}_{n}.png", bbox_inches='tight', pad_inches=0)
        plt.close()  # Ferme la figure proprement
        n+=1

out_name = 'zone6_B4'

generate_temporal_gif_from_h5(
    h5_path='data/dataset.h5',
    zone_name='zone6',
    band_index=2,
    patch_size=(24, 24),
    out_name=out_name ,
    vmin=0,
    vmax=3000)

import imageio.v2 as imageio
import os

# Charger tous les fichiers PNG triés
images = []
n=0
for i in range(12):  # ou range(len(mosaics)) si dynamique
    filename = f"../plots/png/{out_name}_2023_{n}.png"
    if os.path.exists(filename):
        images.append(imageio.imread(filename))
    n+=1
# Créer le GIF
imageio.mimsave("../plots/gif/zone3_B2.gif", images, fps=2)
print("✅ GIF enregistré sous 'mon_animation.gif'")
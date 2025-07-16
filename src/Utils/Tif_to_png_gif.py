import h5py
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import imageio.v2 as imageio

# Colormaps by band index
BAND_COLORMAPS = {
    0: 'Blues',     # B2 - Blue
    1: 'Greens',    # B3 - Green
    2: 'Reds',      # B4 - Red
    3: 'inferno',   # B8 - NIR
}

def get_transform_for_zone(zone_name):
    tif_path = f'data/Tif/{zone_name}/labels_raster_masked.tif'
    with rasterio.open(tif_path) as src:
        return src.transform

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



def generate_band_pngs(h5_path, zone_name, zone_label, band_index, patch_size, output_dir='plots/png'):
    cmap = BAND_COLORMAPS.get(band_index, 'viridis')
    band_name = ['B1', 'B2', 'B3', 'B4'][band_index]
    transform = get_transform_for_zone(zone_name)

    with h5py.File(h5_path, 'r') as f:
        data = f['data'][:]       # (N, T, C, H, W)
        coords = f['coords'][:]   # (N, 2)
        zones = f['zones'][:].astype(str)

    zone_mask = (zones == zone_name)
    if not np.any(zone_mask):
        raise ValueError(f" No data found for zone '{zone_name}'")

    zone_data = data[zone_mask]
    zone_coords = coords[zone_mask]
    time_steps = zone_data.shape[1]

    out_folder = f"{output_dir}/{zone_label}/{band_name}"
    os.makedirs(out_folder, exist_ok=True)

    for t in tqdm(range(time_steps), desc=f"{zone_name} - {band_name}"):
        band_patches = zone_data[:, t, band_index]
        mosaic = create_spatial_mosaic_from_geo_coords(band_patches, zone_coords, patch_size, transform)

        # 🧼 Remove cloud-contaminated or invalid values
        if band_index == 4:
            mosaic[(mosaic > 5000) | (mosaic < 0)] = 0 # clouds / noise
        
        else : 
            mosaic[(mosaic > 4000) | (mosaic < 0)] = 0 # clouds / noise

        
        H  , W = mosaic.shape[:2]
        zoom_factor = 8

        # Taille du patch zoomé
        zoom_h = H // zoom_factor
        zoom_w = W // zoom_factor

        # Exemple : zoomer sur le coin supérieur droit
        zoomed_mosaic = mosaic[0:zoom_h, W - zoom_w:W]

        # 🖼 Plot with proper handling of NaNs
        plt.imshow(zoomed_mosaic, cmap=cmap, vmin=0, vmax=4000)
        plt.title(f"{zone_label} - {band_name} - month - {t+1}")
        plt.axis('off')
        plt.savefig(f"{out_folder}/frame_{t:02d}.png", bbox_inches='tight', pad_inches=0)
        plt.close()


def generate_gif_from_pngs(zone_label, band_index, png_dir='plots/png', gif_dir='plots/gif'):
    """
    Generate a GIF from PNG files for a specific zone and band.
    
    Args:
        zone_name (str): Human-readable zone name (e.g., 'paris')
        band_index (int): Index of the spectral band
        output_dir (str): Path where PNGs are stored
        gif_dir (str): Output path for the resulting GIF
    """
    import imageio.v2 as imageio
    import os

    band_id = f"B{band_index + 1}"
    frame_dir = frame_dir = f"{png_dir}/{zone_label}/{zone_label}/B{band_index + 1}"
    
    # Make sure the frame directory exists
    if not os.path.exists(frame_dir):
        raise FileNotFoundError(f"Frame directory not found: {frame_dir}")

    frames = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])
    images = [imageio.imread(os.path.join(frame_dir, f)) for f in frames]

    os.makedirs(gif_dir, exist_ok=True)
    gif_path = os.path.join(gif_dir, f"{zone_label}_{band_id}.gif")
    imageio.mimsave(gif_path, images, fps=2)
    print(f"✅ GIF saved to '{gif_path}'")



zones_dict = {

    'zone32': "Alsace",

}

  #  'zone1': "Nord-Picardie",
  #  'zone6': "Paris",
  #  'zone37': "Mediterranean",
  #  'zone27':"Massif_Central"

h5_path='manual_jobs/dataset_test.h5'

bands = [0, 1, 2, 3]  # Corresponds to B2, B3, B4, B8

for zone_key, zone_label in zones_dict.items():
    for band in bands:
        generate_band_pngs(
            h5_path=h5_path,
            zone_name=zone_key, 
            zone_label = zone_label,       # Still uses internal zone ID from HDF5
            band_index=band,
            patch_size=(24, 24),
            output_dir=f'plots/png/{zone_label}'  # Uses human-readable name in path
        )
        generate_gif_from_pngs(
            zone_label=zone_label,      # Use label for output path and GIF naming
            band_index=band,
            gif_dir='plots/gif'
        )
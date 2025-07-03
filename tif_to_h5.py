import os
import argparse
import numpy as np
import h5py
import rasterio
from rasterio.windows import Window
from tqdm import tqdm



def mask_no_labeled_pixel(tif_dir):
    with rasterio.open('labels_raster.tif') as src:
        img_reference = src.read(1)  # lire la première bande
    # Get all tif files sorted by date
    tif_files = sorted([os.path.join(tif_dir, f) for f in os.listdir(tif_dir) if f.endswith('.tif')])

    for tif_path in tqdm(tif_files, desc="mask des TIF"):
        with rasterio.open(tif_path) as src:
            data = src.read()  # Shape: (bands, height, width)
            profile = src.profile
            # Masque: True là où la valeur de référence est 0
            mask = (img_reference == 0)
            # Appliquer le masque à toutes les bandes
            data[:, mask] = 0
        # Sauvegarde dans un nouveau fichier
        base_name = os.path.basename(tif_path).replace('.tif', '_masked.tif')
        output_path = os.path.join('data/Tif', base_name)

        # Écriture du fichier masqué
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data)


def extract_patches(image_path, patch_size=(24, 24), stride=24):
    patches = []
    with rasterio.open(image_path) as src:
        width, height = src.width, src.height
        for i in range(0, width - patch_size[0] + 1, stride):
            for j in range(0, height - patch_size[1] + 1, stride):
                window = Window(i, j, *patch_size)
                patch = src.read(window=window)  # shape: (bands, H, W)
                patches.append(patch)
    return patches





def build_dataset(tif_dir, output_path, patch_size=(24, 24), stride=24):

    mask_no_labeled_pixel(tif_dir)

    # Get all tif files sorted by date

    tif_files = sorted([os.path.join('data/Tif', f) for f in os.listdir('data/Tif') if f.endswith('.tif')])
    assert len(tif_files) > 0, "No .tif files found in directory."

    print(f"Found {len(tif_files)} .tif files. Extracting patches...")

    list_of_patch_sets = []
    for tif in tqdm(tif_files, desc="Extracting from .tif files"):
        patches = extract_patches(tif, patch_size, stride)
        list_of_patch_sets.append(patches)

    # Stack patches per spatial location: shape (T, C, H, W)
    stacked_patches = []
    n_patches = len(list_of_patch_sets[0])
    for patch_idx in range(n_patches):
        time_series = [list_of_patch_sets[t][patch_idx] for t in range(len(list_of_patch_sets))]
        stacked = np.stack(time_series, axis=0)  # shape: (T, C, H, W)
        stacked_patches.append(stacked)

    non_empty_mask = np.any(stacked_patches != 0, axis=(1, 2, 3, 4))
    filtered_patches = stacked_patches[non_empty_mask]

    print(f"Saving patch conserved : n= {filtered_patches.shape[0]} patches not null over {stacked_patches.shape[0]} in total)")

    with h5py.File(output_path, 'w') as hf:
        hf.create_dataset("data", data=stacked_patches, compression="gzip")
    print("Done ✅")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .tif time series to .h5 dataset for CNN training.")
    parser.add_argument('--tif_dir', type=str, default= "data/Tif", help="Directory with .tif files (ordered by date)")
    parser.add_argument('--output', type=str, default= "Dataset.h5", help="Output path to .h5 file")
    parser.add_argument('--patch_size', type=int, default=24, help="Patch size (assumes square)")
    parser.add_argument('--stride', type=int, default=24, help="Stride between patches") # default=24=patch_size for Non-overlapping patches

    args = parser.parse_args()

    build_dataset(
        tif_dir=args.tif_dir,
        output_path=args.output,
        patch_size=(args.patch_size, args.patch_size),
        stride=args.stride
    )



########### to execute the code, paste this in your terminal 
# python tif_to_h5.py --tif_dir ./file_to/my_tifs --output dataset.h5 --patch_size 24 --stride 24
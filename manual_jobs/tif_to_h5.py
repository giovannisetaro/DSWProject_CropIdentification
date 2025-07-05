import os
import argparse
import numpy as np
import h5py
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
import re




def extract_patches_with_coords(image_path, patch_size=(24, 24), stride=24):
    patches = []
    coords = []
    with rasterio.open(image_path) as src:
        transform = src.transform
        width, height = src.width, src.height

        for i in range(0, height - patch_size[0] + 1, stride):
            for j in range(0, width - patch_size[1] + 1, stride):
                window = Window(j, i, patch_size[1], patch_size[0])  # Note l'ordre: col, row, width, height
                patch = src.read(window=window)  # shape: (bands, H, W)
                patches.append(patch)

                # Convertir pixel (i,j) en coordonnées géospatiales (x,y)
                x, y = rasterio.transform.xy(transform, i, j)
                coords.append((x, y))

    return patches, coords



def extract_date_from_filename(filename):
    match = re.search(r'(\d{8})', filename)  # Cherche une date au format "YYYYMMDD"
    if match:
        date_str = match.group(1)
        return pd.to_datetime(date_str, format="%Y%m%d")
    return None  # ou "Unknown" si tu préfères






def build_dataset(tif_dir, label_path, output_path, patch_size=(24, 24), stride=24):
    tif_files = sorted([os.path.join(tif_dir, f) for f in os.listdir(tif_dir) if f.endswith('.tif')])
    assert len(tif_files) > 0, "No .tif files found."

    print(f"Found {len(tif_files)} .tif files. Extracting patches...")

    list_of_patch_sets = []
    list_of_dates = []
    all_coords = None

    for tif in tqdm(tif_files, desc="Extracting from .tif files"):
        patches, coords = extract_patches_with_coords(tif, patch_size, stride)
        date = extract_date_from_filename(os.path.basename(tif)) 
        list_of_patch_sets.append(patches)
        list_of_dates.append(date)
        if all_coords is None:
            all_coords = coords  # Same for all timestamps

    # Stack patches along time for each spatial location
    stacked_patches = []
    for idx in range(len(all_coords)):
        time_series = [list_of_patch_sets[t][idx] for t in range(len(tif_files))]  # [T, C, H, W]
        stacked = np.stack(time_series, axis=0)  # [T, C, H, W]
        stacked_patches.append(stacked)
    stacked_patches = np.stack(stacked_patches)  # [N, T, C, H, W]

    # Process label image (assuming it aligns spatially)
    label_dataset = rasterio.open(label_path)
    label_img = label_dataset.read(1)  # [H, W]

    label_patches = []
    for (x, y) in all_coords:
        row, col = rasterio.transform.rowcol(label_dataset.transform, x, y)
        patch = label_img[row:row+patch_size[0], col:col+patch_size[1]]
        label_patches.append(patch)

    label_patches = np.stack(label_patches)
    label_dataset.close()


    # mask empty patches
    non_empty_mask = np.any(stacked_patches != 0, axis=(1, 2, 3, 4))
    filtered_patches = stacked_patches[non_empty_mask]
    filtered_labels = label_patches[non_empty_mask]
    filtered_coords = np.array(all_coords)[non_empty_mask]


    # Converting the list of pd.datetimes into an array to store it in the .h5
    dates_array = np.array(list_of_dates, dtype='datetime64[ns]')
    group.create_dataset("dates", data=dates_array)
    # Save to .h5

    with h5py.File(output_path, 'w') as hf:
        group = hf.create_group("dataset")
        group.create_dataset("data", data=filtered_patches, compression="gzip")      # [N, T, C, H, W]
        group.create_dataset("gt_labels", data=filtered_labels, compression="gzip")     # [N, H, W]
        group.create_dataset("coords", data=filtered_coords, compression="gzip")     # [N, 2] → (x, y)
        group.create_dataset("dates", data=dates_array)                    # [T]

#dataset.h5
#└── dataset/
#    ├── data     [N, T, C, H, W]
#    ├── labels   [N, H, W]
#    ├── coords   [N, 2]
#    └── dates    [T]

# with : for one i = 
# x = hf["dataset/data"][i]       # [T, C, H, W]
# y = hf["dataset/labels"][i]     # [H, W]
# xy = hf["dataset/coords"][i]    # coord (x,y) 

    print(f"✅ Saved {filtered_patches.shape[0]} patches to {output_path}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .tif time series to .h5 dataset for CNN training.")
    parser.add_argument('--tif_dir', type=str, default= "data/Tif", help="Directory with .tif files (ordered by date)")
    parser.add_argument('--output', type=str, default= "data/Dataset.h5", help="Output path to .h5 file")
    parser.add_argument('--patch_size', type=int, default=24, help="Patch size (assumes square)")
    parser.add_argument('--stride', type=int, default=24, help="Stride between patches") # default padding=24=patch_size for Non-overlapping patches

    args = parser.parse_args()

    build_dataset(
        tif_dir=args.tif_dir,
        output_path=args.output,
        patch_size=(args.patch_size, args.patch_size),
        stride=args.stride
    )



########### to execute the code, paste this in your terminal 
# python tif_to_h5.py --tif_dir ./file_to/my_tifs --output dataset.h5 --patch_size 24 --stride 24
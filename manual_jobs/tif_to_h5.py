import os
import argparse
import numpy as np
import h5py
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
import re
import pandas as pd 
import warnings





def extract_patches_with_coords(image_path, patch_size=(24, 24), stride=24):
    patches = []
    coords = []
    #nan_counts = []
    #fully_nan_count = 0
    with rasterio.open(image_path) as src:
        transform = src.transform
        width, height = src.width, src.height

        for i in range(0, height - patch_size[0] + 1, stride):
            for j in range(0, width - patch_size[1] + 1, stride):
                window = Window(j, i, patch_size[1], patch_size[0])  # Note l'ordre: col, row, width, height
                patch = src.read(window=window)  # shape: (bands, H, W)
                patches.append(patch)

                # Compter les NaN dans le patch (tous canaux confondus)
                nan_count = np.isnan(patch).sum()
                #nan_counts.append(nan_count)
                                # Vérifier si tout le patch est NaN
                #if np.isnan(patch).all():
                #    fully_nan_count += 1

                
                # Convertir pixel (i,j) en coordonnées géospatiales (x,y)
                x, y = rasterio.transform.xy(transform, i, j) # upper left corner of the patch
                coords.append((x, y))
        #print(f"Patchs complètement NaN : {fully_nan_count}")
        #print(f"Nombre de patchs extraits pour l'image {image_path} : {len(patches)}")
        #print(f"Nombre total de NaN dans tous les patchs de cette image(a travers tt les bandes) : {np.sum(nan_counts)}")

    return patches, coords



def extract_date_from_filename(filename):
    match = re.search(r'(\d{8})', filename)
    if match:
        return match.group(1)  # chaîne 'YYYYMMDD'
    return None




def process_zone(zone_dir, patch_size=(24, 24), stride=24):
    # Récupérer tous les .tif dans la zone (excepté labels)
    tif_files = sorted([os.path.join(zone_dir, f) for f in os.listdir(zone_dir)
                        if f.endswith('.tif') and 'labels' not in f])
    if len(tif_files) == 0:
        warnings.warn(f"No .tif files found in {zone_dir}. Skipping this zone.")
        return None, None, None, None

    print(f"Processing zone {zone_dir}, found {len(tif_files)} images")

    list_of_patch_sets = []
    list_of_dates = []
    all_coords = None

    for tif in tqdm(tif_files, desc=f"Extracting patches in {os.path.basename(zone_dir)}"):
        patches, coords = extract_patches_with_coords(tif, patch_size, stride)
        date = extract_date_from_filename(os.path.basename(tif))
        list_of_patch_sets.append(patches)
        list_of_dates.append(date)
        if all_coords is None:
            all_coords = coords
    print(f"list of patch set for zone {zone_dir} =",len(list_of_patch_sets))

    # Empiler le temps pour chaque patch spatial
    stacked_patches = []
    for idx in range(len(all_coords)):
        time_series = [list_of_patch_sets[t][idx] for t in range(len(tif_files))]
        stacked = np.stack(time_series, axis=0)  # (T, C, H, W)
        stacked_patches.append(stacked)
    stacked_patches = np.stack(stacked_patches)  # (N, T, C, H, W)

    # Traiter le raster des labels
    label_path = os.path.join(zone_dir, 'labels_raster_masked.tif')
    with rasterio.open(label_path) as label_dataset:
        label_img = label_dataset.read(1)
        Id_Parcelles_img = label_dataset.read(2)
        label_patches = []
        Id_Parcelles_patches = []
        for (x, y) in all_coords:
            row, col = rasterio.transform.rowcol(label_dataset.transform, x, y)
            Label_patch = label_img[row:row+patch_size[0], col:col+patch_size[1]]
            ID_Parcelle_patch = Id_Parcelles_img[row:row+patch_size[0], col:col+patch_size[1]]
            label_patches.append(Label_patch)
            Id_Parcelles_patches.append(ID_Parcelle_patch)
        label_patches = np.stack(label_patches)
        Id_Parcelles_patches = np.stack(Id_Parcelles_patches)

    # Masquer les patches vides
    non_empty_mask = np.any(stacked_patches != 0, axis=(1, 2, 3, 4))
    print("Nombre de patchs conservés :", np.sum(non_empty_mask), "sur", len(stacked_patches))
    filtered_patches = stacked_patches[non_empty_mask]
    filtered_labels = label_patches[non_empty_mask]
    filtered_ID_Parcelles = Id_Parcelles_patches[non_empty_mask]
    filtered_coords = np.array(all_coords)[non_empty_mask]

    # Dates en np.datetime64 (même pour tous les patches)
    dates_array = np.array(list_of_dates, dtype='datetime64[ns]')


    return filtered_patches, filtered_labels,filtered_ID_Parcelles, filtered_coords, dates_array

def build_all_zones_dataset(data_dir, output_path, patch_size=(24, 24), stride=24):
    zones = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print(f"Found zones: {[os.path.basename(z) for z in zones]}")

    all_data = []
    all_labels = []
    all_ID_Parcelles = []
    all_coords = []
    dates_ref = None

    for zone_dir in zones:
        data, labels,ID_Parcelles, coords, dates = process_zone(zone_dir, patch_size, stride)

        if dates_ref is None:
            dates_ref = dates  # récupérer la référence une fois

        all_data.append(data)
        all_labels.append(labels)
        all_ID_Parcelles.append(ID_Parcelles)
        all_coords.append(coords)

    dates_str = [str(d) for d in dates_ref if d is not None]

    # Concaténer tous les patchs des zones
    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_ID_Parcelles = np.concatenate(all_ID_Parcelles, axis=0)
    all_coords = np.concatenate(all_coords, axis=0)

    # mask to remove NaN
    valid_mask = ~np.isnan(all_data).any(axis=(1, 2, 3, 4))  # (N, T, C, H, W) 

    print(f"Patchs valides : {np.sum(valid_mask)} / {len(all_data)}")

    all_data = all_data[valid_mask]
    all_labels = all_labels[valid_mask]
    all_ID_Parcelles = all_ID_Parcelles[valid_mask]
    all_coords = all_coords[valid_mask]

    # Sauvegarder dans un fichier h5 plat
    with h5py.File(output_path, 'w') as hf:
        hf.create_dataset("data", data=all_data.astype(np.float32), compression="gzip")
        hf.create_dataset("labels", data=all_labels.astype(np.float32), compression="gzip")
        hf.create_dataset("ID_Parcelles", data=all_ID_Parcelles.astype(np.float32), compression="gzip")
        hf.create_dataset("coords", data=all_coords, compression="gzip")
        dt = h5py.string_dtype(encoding='utf-8')
        hf.create_dataset("dates", data=np.array(dates_str, dtype=dt))

    print(f"✅ Saved flat dataset with {all_data.shape[0]} patches to {output_path}")
#dataset.h5
#└── dataset/
#    ├── data     [N, T, C, H, W]
#    ├── labels   [N, H, W]
#    ├── ID_Parcelles   [N, H, W]
#    ├── coords   [N, 2]
#    └── dates    [T]

# with : for one i = 
# x = hf["dataset/data"][i]       # [T, C, H, W]
# y = hf["dataset/labels"][i]     # [H, W]
# xy = hf["dataset/coords"][i]    # coord (x,y) 






if __name__ == "__main__":
    

    parser = argparse.ArgumentParser(description="Convert multiple zones of .tif time series to a single flat .h5 dataset for CNN training.")
    parser.add_argument('--data_dir', type=str, default="data/Tif", help="Root directory containing subfolders per zone")
    parser.add_argument('--output', type=str, default="data/Dataset.h5", help="Output path to the flat .h5 file")
    parser.add_argument('--patch_size', type=int, default=24, help="Patch size (assumes square)")
    parser.add_argument('--stride', type=int, default=24, help="Stride between patches")

    args = parser.parse_args()

    build_all_zones_dataset(
        data_dir=args.data_dir,
        output_path=args.output,
        patch_size=(args.patch_size, args.patch_size),
        stride=args.stride
    )



########### to execute the code, paste this in your terminal 
# python tif_to_h5.py --tif_dir ./file_to/my_tifs --output dataset.h5 --patch_size 24 --stride 24
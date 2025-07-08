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



def interpolate_nan_along_time(tensor):
    """
    Interpolates missing (NaN) values along the time axis for each pixel (C, H, W) independently.
    
    Parameters:
    tensor (np.ndarray): Input tensor of shape (T, C, H, W)

    Returns:
    np.ndarray: Tensor with NaNs replaced by temporal interpolation
    """
    T, C, H, W = tensor.shape
    tensor_interp = tensor.copy()

    total = C * H * W
    with tqdm(total=total, desc="Interpolating NaNs", leave=False) as pbar:
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    ts = tensor[:, c, h, w]
                    if np.isnan(ts).any():
                        valid_idx = np.where(~np.isnan(ts))[0]
                        if len(valid_idx) == 0:
                            pass  # Leave as NaN
                        elif len(valid_idx) == 1:
                            tensor_interp[:, c, h, w] = ts[valid_idx[0]]
                        else:
                            for t in range(T):
                                if np.isnan(ts[t]):
                                    before = valid_idx[valid_idx < t]
                                    after = valid_idx[valid_idx > t]
                                    if before.size > 0 and after.size > 0:
                                        t_before = before[-1]
                                        t_after = after[0]
                                        val = (ts[t_before] + ts[t_after]) / 2.0
                                    elif after.size > 0:
                                        val = ts[after[0]]
                                    elif before.size > 0:
                                        val = ts[before[-1]]
                                    else:
                                        continue
                                    tensor_interp[t, c, h, w] = val
                    pbar.update(1)
    return tensor_interp


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
    list_of_zones = []
    all_coords = None

    for tif in tqdm(tif_files, desc=f"Extracting patches in {os.path.basename(zone_dir)}"):
        patches, coords = extract_patches_with_coords(tif, patch_size, stride)
        list_of_patch_sets.append(patches)
        if all_coords is None:
            all_coords = coords
    print(f"list of patch set for zone {zone_dir} =",len(list_of_patch_sets))

    # Empiler le temps pour chaque patch spatial
    stacked_patches = []

    for idx in range(len(all_coords)):
        list_of_zones.append(os.path.basename(zone_dir))

        date = extract_date_from_filename(os.path.basename(tif))
        list_of_dates.append(date)

        time_series = [list_of_patch_sets[t][idx] for t in range(len(tif_files))]

        stacked = np.stack(time_series, axis=0)  # (T, C, H, W)
        stacked = interpolate_nan_along_time(stacked)
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
    filtered_zones = np.array(list_of_zones)[non_empty_mask]
    filtered_dates = np.array(list_of_dates)[non_empty_mask]

   


    return filtered_patches, filtered_labels,filtered_ID_Parcelles, filtered_coords, filtered_dates, filtered_zones

def build_all_zones_dataset(data_dir, output_path, patch_size=(24, 24), stride=24):
    zones = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print(f"Found zones: {[os.path.basename(z) for z in zones]}")

    all_data = []
    all_labels = []
    all_ID_Parcelles = []
    all_coords = []
    all_zone = []
    all_dates = []
    

    for zone_dir in zones:
        data, labels,ID_Parcelles, coords, dates, zones_ref  = process_zone(zone_dir, patch_size, stride)


        all_data.append(data)
        all_labels.append(labels)
        all_ID_Parcelles.append(ID_Parcelles)
        all_coords.append(coords)
        all_zone.append(zones_ref)
        all_dates.append(dates)
  

    
    # Concaténer tous les patchs des zones
    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_ID_Parcelles = np.concatenate(all_ID_Parcelles, axis=0)
    all_coords = np.concatenate(all_coords, axis=0)
    all_zone = np.concatenate(all_zone, axis=0)
    all_dates = np.concatenate(all_dates, axis=0)

    # mask to remove NaN
    valid_mask = ~np.isnan(all_data).any(axis=(1, 2, 3, 4))  # (N, T, C, H, W) ###################### faut changer ça c'est pas top

    print(f"Patchs valides : {np.sum(valid_mask)} / {len(all_data)}") 

    all_data = all_data[valid_mask]
    all_labels = all_labels[valid_mask]
    all_ID_Parcelles = all_ID_Parcelles[valid_mask]
    all_coords = all_coords[valid_mask]
    all_zone = all_zone[valid_mask]
    all_dates = all_dates[valid_mask]
    

    # Sauvegarder dans un fichier h5 plat
    with h5py.File(output_path, 'w') as hf:
        hf.create_dataset("data", data=all_data.astype(np.float32), compression="gzip")
        hf.create_dataset("labels", data=all_labels.astype(np.float32), compression="gzip")
        hf.create_dataset("ID_Parcelles", data=all_ID_Parcelles.astype(np.float32), compression="gzip")
        hf.create_dataset("coords", data=all_coords, compression="gzip")
        hf.create_dataset("dates", data=np.array(all_dates, dtype = h5py.string_dtype(encoding='utf-8')))
        hf.create_dataset("zones", data=np.array(all_zone, dtype = h5py.string_dtype(encoding='utf-8')))

    print(f"✅ Saved flat dataset with {all_data.shape[0]} patches to {output_path}")
#dataset.h5
#└── 
#    ├── data     [N, T, C, H, W]
#    ├── labels   [N, H, W]
#    ├── ID_Parcelles   [N, H, W]
#    ├── coords   [N, 2]
#    ├── zones    [N, 1]
#    └── dates    [N, 1]

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
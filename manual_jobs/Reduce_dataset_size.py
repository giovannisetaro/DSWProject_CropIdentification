import h5py
import numpy as np

# Zones to remove 
target_zones = [
    "zone27", "zone28", "zone29", "zone30", "zone31",
    "zone32", "zone33", "zone34", "zone35", "zone36",
    "zone37", "zone38", "zone39", "zone40", "zone41"
]


def process_dataset(input_path, output_path, target_zones):
    print(f"Processing {input_path}...")
    with h5py.File(input_path, 'r') as f:
        all_data = f['data'][:]         # [N, T, C, H, W]
        all_labels = f['labels'][:]     # [N, H, W]
        all_ID_Parcelles = f['ID_Parcelles'][:]  # [N, H, W]
        all_coords = f['coords'][:]     # [N, 2]
        all_zones = f['zones'][:].astype(str).flatten()  # [N,]
        all_dates = f['dates'][:].astype(str).flatten()  # [N,]

    # mask 
    mask_target = np.isin(all_zones, target_zones)

    # Saving the sub samples
    with h5py.File(output_path, 'w') as hf:
        hf.create_dataset("data", data=all_data[mask_target].astype(np.float32), compression="gzip")
        hf.create_dataset("labels", data=all_labels[mask_target].astype(np.float32), compression="gzip")
        hf.create_dataset("ID_Parcelles", data=all_ID_Parcelles[mask_target].astype(np.float32), compression="gzip")
        hf.create_dataset("coords", data=all_coords[mask_target], compression="gzip")
        hf.create_dataset("dates", data=np.array(all_dates[mask_target], dtype=h5py.string_dtype('utf-8')))
        hf.create_dataset("zones", data=np.array(all_zones[mask_target], dtype=h5py.string_dtype('utf-8')))
    print(f"{output_path} saved with number of mask = {np.sum(mask_target)} .\n")

# datasets to process paths :
datasets = [
    ("../data/dataset_test.h5", "../data/dataset_test_litle.h5"),
    ("../data/dataset_val_train.h5", "../data/dataset_train_val_litle.h5")
]

# run the masking and saving task 
for input_path, output_path in datasets:
    process_dataset(input_path, output_path, target_zones)



import h5py
import numpy as np

# Zones to remove 
target_zones = [
    "zone27", "zone28", "zone29", "zone30", "zone31",
    "zone32", "zone33", "zone34", "zone35", "zone36",
    "zone37", "zone38", "zone39", "zone40", "zone41"
]


def process_dataset(input_path, output_path, target_zones):
    print(f"Processing {input_path}...")
    with h5py.File(input_path, 'r') as f:
        all_data = f['data'][:]         # [N, T, C, H, W]
        all_labels = f['labels'][:]     # [N, H, W]
        all_ID_Parcelles = f['ID_Parcelles'][:]  # [N, H, W]
        all_coords = f['coords'][:]     # [N, 2]
        all_zones = f['zones'][:].astype(str).flatten()  # [N,]
        all_dates = f['dates'][:].astype(str).flatten()  # [N,]

    # mask 
    mask_target = np.isin(all_zones, target_zones)

    # Saving the sub samples
    with h5py.File(output_path, 'w') as hf:
        hf.create_dataset("data", data=all_data[mask_target].astype(np.float32), compression="gzip")
        hf.create_dataset("labels", data=all_labels[mask_target].astype(np.float32), compression="gzip")
        hf.create_dataset("ID_Parcelles", data=all_ID_Parcelles[mask_target].astype(np.float32), compression="gzip")
        hf.create_dataset("coords", data=all_coords[mask_target], compression="gzip")
        hf.create_dataset("dates", data=np.array(all_dates[mask_target], dtype=h5py.string_dtype('utf-8')))
        hf.create_dataset("zones", data=np.array(all_zones[mask_target], dtype=h5py.string_dtype('utf-8')))
    print(f"{output_path} saved with number of mask = {np.sum(mask_target)} .\n")

# datasets to process paths :
datasets = [
    ("../data/dataset_test.h5", "../data/dataset_test_litle.h5"),
    ("../data/dataset_val_train.h5", "../data/dataset_train_val_litle.h5")
]

# run the masking and saving task 
for input_path, output_path in datasets:
    process_dataset(input_path, output_path, target_zones)
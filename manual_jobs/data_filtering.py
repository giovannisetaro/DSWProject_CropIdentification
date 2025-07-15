import h5py
import numpy as np

# Zones to remove from the dataset
target_zones = [
    "zone27", "zone28", "zone29", "zone30", "zone31",
    "zone32", "zone33", "zone34", "zone35", "zone36",
    "zone37", "zone38", "zone39", "zone40", "zone41"
]

def process_dataset(input_path, output_path, target_zones):
    print(f"Processing {input_path}...")
    
    # Open the input HDF5 file and read all relevant datasets
    with h5py.File(input_path, 'r') as f:
        all_data = f['data'][:]              # Shape: [N, T, C, H, W]
        all_labels = f['labels'][:]          # Shape: [N, H, W]
        all_ID_Parcelles = f['ID_Parcelles'][:]  # Shape: [N, H, W]
        all_coords = f['coords'][:]          # Shape: [N, 2]
        # Convert zones and dates to strings and flatten arrays to 1D
        all_zones = f['zones'][:].astype(str).flatten()  
        all_dates = f['dates'][:].astype(str).flatten()  

    # Create boolean mask selecting samples with zones in target_zones
    mask_target = np.isin(all_zones, target_zones)

    # Save filtered data into a new HDF5 file
    with h5py.File(output_path, 'w') as hf:
        hf.create_dataset("data", data=all_data[mask_target].astype(np.float32), compression="gzip")
        hf.create_dataset("labels", data=all_labels[mask_target].astype(np.float32), compression="gzip")
        hf.create_dataset("ID_Parcelles", data=all_ID_Parcelles[mask_target].astype(np.float32), compression="gzip")
        hf.create_dataset("coords", data=all_coords[mask_target], compression="gzip")
        # Save dates and zones as UTF-8 strings
        hf.create_dataset("dates", data=np.array(all_dates[mask_target], dtype=h5py.string_dtype('utf-8')))
        hf.create_dataset("zones", data=np.array(all_zones[mask_target], dtype=h5py.string_dtype('utf-8')))

    print(f"{output_path} saved with {np.sum(mask_target)} samples.\n")


# List of input/output dataset file paths to process
datasets = [
    ("/storage/courses/CropIdentification/DSWProject_CropIdentification/data/dataset_test.h5",
     "/storage/courses/CropIdentification/DSWProject_CropIdentification/data/dataset_test_litle.h5"),
    ("/storage/courses/CropIdentification/DSWProject_CropIdentification/data/dataset_val_train.h5",
     "/storage/courses/CropIdentification/DSWProject_CropIdentification/data/dataset_train_val_litle.h5")
]

# Run the processing on each dataset pair
for input_path, output_path in datasets:
    process_dataset(input_path, output_path, target_zones)

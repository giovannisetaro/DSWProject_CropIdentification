import h5py
import numpy as np
import os

def split_hdf5_dataset(input_path, output_dir, num_splits=8):
    os.makedirs(output_dir, exist_ok=True)

    # Open the input HDF5 file
    with h5py.File(input_path, 'r') as f:
        x = f['data']            # Dataset features
        y = f['labels']          # Corresponding labels
        num_samples = x.shape[0]
        num_classes = int(y[:].max()) + 1  # Assuming labels go from 0 to N-1

        print(f"Total samples: {num_samples}")
        print(f"Unique classes: {len(np.unique(y[:]))}")
        print(f"Maximum label: {y[:].max()} --> num_classes = {num_classes}")

        # Split indices into equal parts
        indices = np.array_split(np.arange(num_samples), num_splits)

        for i, split_idx in enumerate(indices):
            output_path = os.path.join(output_dir, f"dataset_split_{i}.h5")
            with h5py.File(output_path, 'w') as out_f:
                # Save the corresponding subset of data and labels
                out_f.create_dataset('data', data=x[split_idx], compression="gzip")
                out_f.create_dataset('labels', data=y[split_idx], compression="gzip")
            print(f"Saved: {output_path} with {len(split_idx)} samples")

    print("Splitting complete.")
    print(f"Files saved to: {output_dir}")

if __name__ == "__main__":
    # Modify these paths if needed
    input_h5_path = "data/dataset_val_train.h5"
    output_directory = "data/split_datasets"
    num_files = 8  # Number of output parts

    split_hdf5_dataset(input_h5_path, output_directory, num_files)

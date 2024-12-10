import h5py
import numpy as np
import os

DATA_DIR = 'data'
#all h5 files startign with split_data_
h5_files = [f for f in os.listdir(DATA_DIR) if f.startswith('split_data_')]
print(f'Found {len(h5_files)} HDF5 files')

for ii, h5_file in enumerate(h5_files):
    print(f'Processing split {ii}')
    with h5py.File(f'data/{h5_file}', 'r') as infile:
        # Iterate over all datasets in the current HDF5 file
        dataset = infile['code']
        dataset_shape = dataset.shape
        print(f'shape: {dataset_shape}')
    LEN_DATA = dataset_shape[0]
    print(f'LEN_DATA: {LEN_DATA}')

    # Create a random permutation of the indices
    indices = np.arange(LEN_DATA)
    indices = np.random.permutation(indices)
    with h5py.File(f'data/split_data_{ii}.h5', 'r') as infile:
        with h5py.File(f'data/shuffled_data_{ii}.h5', 'w') as outfile:
            for dataset_name in infile.keys():
                dataset = infile[dataset_name]
                #turn into numpy array
                dataset = np.array(dataset)
                shuffled_dataset = dataset[indices]
                print(f'shuffled_dataset: {shuffled_dataset.shape}')
                outfile.create_dataset(dataset_name, data=shuffled_dataset, chunks=True, maxshape=(None, *dataset.shape[1:]))

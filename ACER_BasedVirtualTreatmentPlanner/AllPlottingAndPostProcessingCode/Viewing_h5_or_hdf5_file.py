import h5py

# Open the HDF5 file
# file = h5py.File("/media/mainul/Chi-Drives/128641-3-E21/var/lib/docker/overlay2/8fa0092f3bc6971c1752f734f8fd34c782377f383ffe6f5a55629a01d3b0f185/diff/home/exx/dose_deposition_full/prostate_dijs/f_dijs/008.hdf5", "r")
file = h5py.File('/media/mainul/Chi-Drives/128641-3-E21/var/lib/docker/overlay2/8fa0092f3bc6971c1752f734f8fd34c782377f383ffe6f5a55629a01d3b0f185/diff/home/exx/dose_deposition_full/plostate_dijs/f_masks/008.h5', "r")
# Define a function to recursively explore the HDF5 file structure
def explore_hdf5_group(group, indent=""):
    for name, item in group.items():
        if isinstance(item, h5py.Group):
            print(f"{indent}Group: {name}")
            explore_hdf5_group(item, indent + "  ")
        elif isinstance(item, h5py.Dataset):
            print(f"{indent}Dataset: {name}")

# Start exploring the file structure from the root group
explore_hdf5_group(file)

# Close the HDF5 file when done
file.close()

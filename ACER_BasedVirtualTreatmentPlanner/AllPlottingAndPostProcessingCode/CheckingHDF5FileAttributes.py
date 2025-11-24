import h5py

# Open the HDF5 file for reading
hdf5_file = h5py.File('001.hdf5', 'r')

# Get all attributes of the HDF5 file
attributes = hdf5_file.attrs

# Iterate through the attributes and print their names and values
for key, value in attributes.items():
    print(f"Attribute: {key}, Value: {value}")

# Close the HDF5 file when you're done
hdf5_file.close()

import scipy as sp
import glob
import re
import numpy as np
import h5py

"""
This is the file to create structure mask using Voxel list files in PROSTATE folder

author: Parth Vijaykumar Soni, pxs3648@mavs.uta.edu 
In case if you find any bugs or inconsistency in the code, feel free to contact.
"""


file_name = "002_structure_mask.h5"
hdf5_file = h5py.File(file_name, 'a')
group_name = 'oar_ptvs'
group = hdf5_file.create_group(group_name)

depth = 90 
rows = 184
cols = 184

total_voxel = depth*rows*cols

# Created list of voxel indices list for entire body
voxel_vector = np.arange(1, total_voxel + 1).reshape(-1,1)

# Iterating over entire VOLIST for Organs at Risk (OARs) and Planning Target Volume (PTVs)
for fname in glob.glob("./PROSTATE/*_VOILIST.mat"): 

    # Used regex to find the organ or ptv name
    pattern = r"(?<=\.\/PROSTATE\\).*?(?=_VOILIST\.mat)"
    match = re.search(pattern, fname)
    if match:
        extracted_string = match.group()
        structure_mask = sp.io.loadmat(fname)
        intersection = np.intersect1d(voxel_vector, structure_mask['v'])

        #creating binary vector
        result = np.isin(voxel_vector, intersection).astype(int)
        print(f"{extracted_string}",np.sum(result), len(result))

        #Reshaping it according to CERR co-ordinate system
        result = result.reshape((depth, cols, rows))
        result = np.transpose(result, (0,2,1))
        group.create_dataset(f"{extracted_string}", data = result)



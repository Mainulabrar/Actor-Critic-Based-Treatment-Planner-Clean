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


file_name = "test_structure_mask.h5"
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
    print("fname",fname)

    # Used regex to find the organ or ptv name
    pattern = r'([^/]+)\.mat$'
    match = re.search(pattern, fname)
    print("match = re.search(pattern, fname)",match)
    
    if match:
        extracted_string = match.group()
        print("extracted_string = match.group()",extracted_string)
        structure_mask = sp.io.loadmat(fname)
        print("structure_mask = sp.io.loadmat(fname)",structure_mask)
        intersection = np.intersect1d(voxel_vector, structure_mask['v'])
        print("intersection = np.intersect1d(voxel_vector, structure_mask['v'])",intersection)

        #creating binary vector
        result = np.isin(voxel_vector, intersection).astype(int)
        print("result = np.isin(voxel_vector, intersection).astype(int)",result)
        print(f"{extracted_string}",np.sum(result), len(result))
        
        #Reshaping it according to CERR co-ordinate system
        result = result.reshape((depth, cols, rows))
        print("result = result.reshape((depth, cols, rows))",result)
        result = np.transpose(result, (0,2,1))
        print("result = np.transpose(result, (0,2,1))",result)
        group.create_dataset(f"{extracted_string}", data = result)



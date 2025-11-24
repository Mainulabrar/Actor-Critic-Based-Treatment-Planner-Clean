import numpy as np
import scipy as sp
import glob
import re
import h5py
from scipy.sparse import csr_matrix, hstack
"""
This is the file to create Dij files in PROSTATE folder

author: Parth Vijaykumar Soni, pxs3648@mavs.uta.edu 
In case if you find any bugs or inconsistency in the code, feel free to contact.
"""
file_name = "002.hdf5"
hdf5_file = h5py.File(file_name, 'a')
group_name = 'Dij'
group = hdf5_file.create_group(group_name)

# iterating over D.mat and BEAMINFO.mat file to extract Dij matrix for each angle and storing them in hdf5 file format
for data_file, inf_file in zip(glob.glob("./PROSTATE/Gantry*_Couch*_D.mat"), glob.glob("./PROSTATE/Gantry*_Couch*_BEAMINFO.mat")):
    pattern = r"Gantry(\d+)"
    match = re.search(pattern, data_file)
    if match:
        Dij = sp.io.loadmat(data_file)
        information = sp.io.loadmat(inf_file)
        gantry_angle = int(match.group(1))
        subgroup_name = f'{gantry_angle:03}'
        Dij_normal = Dij['D'].transpose()
        Dij_new = csr_matrix(Dij_normal)

        # # Desired number of columns
        # desired_num_columns = 190

        # # Calculate the number of columns to add
        # num_columns_to_add = desired_num_columns - Dij['D'].shape[1]

        # if num_columns_to_add > 0:
        # # Create new columns filled with zeros
        #     new_columns = csc_matrix(np.zeros((Dij['D'].shape[0], num_columns_to_add)))

        #     # Stack the new columns horizontally to the existing matrix
        #     Dij_new = hstack([Dij['D'], new_columns])
        # else:
        #     Dij_new = Dij['D']


        # Extracting the data, indices and indptr from Dij sparse matrix which is in the csc form.
        data = Dij_new.data
        indices = Dij_new.indices
        indptr = Dij_new.indptr

        # Getting beamlet number form BEAMINFO.mat file
        beamlet_num = information['numBeamlets'][0][0]
        subgroup = group.create_group(subgroup_name)

        # Creating dataset in subgroup 'Dij' for all the angles which are there in PROSTATE folder, and using appropriate parameters to compress it into small size
        subgroup.create_dataset('data',  data = data, maxshape = data.shape, compression = 'gzip', compression_opts = 9, dtype = h5py.h5t.IEEE_F32LE)
        subgroup.create_dataset('indices', data = indices, maxshape = indices.shape, compression = 'gzip', compression_opts = 9, dtype=h5py.h5t.STD_I32LE)
        subgroup.create_dataset('indptr', data = indptr, maxshape = indptr.shape, compression = 'gzip', compression_opts = 9, dtype=h5py.h5t.STD_I64LE )
        
        # Creating attributes to get extract the information about the Dij matrix during run time
        subgroup.attrs.create("h5sparse_shape", data = Dij_new.shape)
        subgroup.attrs.create("h5sparse_format", data = "csr")

        


        





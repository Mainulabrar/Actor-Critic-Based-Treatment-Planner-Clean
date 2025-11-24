import scipy as sp
import numpy as np
import h5py
"""
This is the file to create dose mask using Dij files in PROSTATE folder

author: Parth Vijaykumar Soni, pxs3648@mavs.uta.edu 
In case if you find any bugs or inconsistency in the code, feel free to contact.
"""


final_mask = []
depth = 90 
rows = 184
cols = 184
total_voxel = depth*rows*cols
voxel_vector = np.arange(1, total_voxel + 1).reshape(-1,1)

# Loop through all the angles to find the idices of voxel affected in each angle
for i in range(0,360, 2):

    Dij = sp.io.loadmat(f"/home/mainul1/PROSTATE/Gantry{i}_Couch0_D.mat")
    information = sp.io.loadmat(f"/home/mainul1/PROSTATE/Gantry{i}_Couch0_BEAMINFO.mat")

    data = Dij['D'].data
    indices = Dij['D'].indices
    indptr = Dij['D'].indptr
    
    # finding out the number of beamlet, because number for beamlet for every angle is different
    num_beamlets = information['numBeamlets'][0][0]
    arr = sp.sparse.csc_array((data, indices, indptr), shape = (3047040,num_beamlets))

    #getting the non-zero indices
    non_zero_idx = arr.nonzero()[0]

    #finding the unique number of voxel which are affected by the beamlet
    idx, counts = np.unique(non_zero_idx, return_counts=True)

    #creating the dose_mask for each angle
    dose_mask = np.intersect1d(voxel_vector, idx).astype(int)

    #union with the final_mask, final_mask will contain the unique indices with has all the affected voxel in each angle, this will be updated in each iteration 
    final_mask = np.union1d(final_mask, dose_mask)

    print(f"No of non-zero index in final_mask after {i} th Gantry angle is {np.sum(np.isin(voxel_vector, final_mask))}")


print("No of voxel affected by beamlet", final_mask.shape[0])

#converting the final_mask into binary vector
result = np.isin(voxel_vector, final_mask).astype(int)

file_name = "test_data_mask.h5"
hdf5_file = h5py.File(file_name, 'a')

#reshaping the mask according to CERR coordinate system
result = result.reshape((depth, cols, rows))
result = np.transpose(result, (0,2,1))
# result = result.reshape(-1,1)
hdf5_file.create_dataset('dose_mask', data = result)



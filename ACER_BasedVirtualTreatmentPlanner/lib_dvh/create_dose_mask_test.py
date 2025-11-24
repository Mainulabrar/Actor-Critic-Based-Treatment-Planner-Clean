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
list1 = [0,32]
# 0,32,64,96, 296, 264, 328

# Loop through all the angles to find the idices of voxel affected in each angle
for i in range(len(list1)):

    Dij = sp.io.loadmat(f"/home/mainul1/PROSTATE/Gantry{list1[i]}_Couch0_D.mat")
    # print(f'Dij = sp.io.loadmat(f"/home/mainul1/PROSTATE/Gantry{list1[i]}_Couch0_D.mat")\n\n', Dij)
    information = sp.io.loadmat(f"/home/mainul1/PROSTATE/Gantry{list1[i]}_Couch0_BEAMINFO.mat")
    print('information = sp.io.loadmat(f"./PROSTATE/Gantry{i}_Couch0_BEAMINFO.mat")\n\n',information)
    
    
    print(f"/home/mainul1/PROSTATE/Gantry{list1[i]}_Couch0_D.mat")
    file = Dij['D']
    # print("list(file.keys()): ",(file.keys()))
    
    print("Dij['D'].shape",Dij['D'].shape)
    
    # print("Dij['D'].data.shape",Dij['D'].data.shape)
    # print("indices = Dij['D'].indices.shape",Dij['D'].indices.shape)
    # print("indptr = Dij['D'].indptr.shape",Dij['D'].indptr.shape)
    # print("Dij['D'].data",Dij['D'].shape)
    print("\n\n")

    data = Dij['D'].data
    indices = Dij['D'].indices
    indptr = Dij['D'].indptr
    
    # finding out the number of beamlet, because number for beamlet for every angle is different
    num_beamlets = information['numBeamlets'][0][0]
    print("num_beamlets = information['numBeamlets'][0][0]\n",num_beamlets)
    arr = sp.sparse.csc_array((data, indices, indptr), shape = (3047040,num_beamlets))
    print("arr = sp.sparse.csc_array((data, indices, indptr), shape = (3047040,num_beamlets))\n", arr)

    #getting the non-zero indices
    non_zero_idx = arr.nonzero()[0]
    print("non_zero_idx = arr.nonzero()[0]\n",non_zero_idx)

    #finding the unique number of voxel which are affected by the beamlet
    idx, counts = np.unique(non_zero_idx, return_counts=True)
    print("idx, counts = np.unique(non_zero_idx, return_counts=True): idx\n",idx)
    print("dx, counts = np.unique(non_zero_idx, return_counts=True): counts\n",counts)

    #creating the dose_mask for each angle
    dose_mask = np.intersect1d(voxel_vector, idx).astype(int)
    print("dose_mask = np.intersect1d(voxel_vector, idx).astype(int)\n", dose_mask)

    #union with the final_mask, final_mask will contain the unique indices with has all the affected voxel in each angle, this will be updated in each iteration 
    final_mask = np.union1d(final_mask, dose_mask)
    print("final_mask = np.union1d(final_mask, dose_mask)\n", final_mask)

    print(f"No of non-zero index in final_mask after {i} th Gantry angle is {np.sum(np.isin(voxel_vector, final_mask))}")


print("No of voxel affected by beamlet", final_mask.shape[0])

#converting the final_mask into binary vector
result = np.isin(voxel_vector, final_mask).astype(int)
print("result = np.isin(voxel_vector, final_mask).astype(int)",result)

file_name = "test.h5"
hdf5_file = h5py.File(file_name, 'a')


#reshaping the mask according to CERR coordinate system
result = result.reshape((depth, cols, rows))
print("result = result.reshape((depth, cols, rows))\n", result)
result = np.transpose(result, (0,2,1))
print("result = np.transpose(result, (0,2,1))\n", result)
# result = result.reshape(-1,1)
hdf5_file.create_dataset('dose_mask', data = result)




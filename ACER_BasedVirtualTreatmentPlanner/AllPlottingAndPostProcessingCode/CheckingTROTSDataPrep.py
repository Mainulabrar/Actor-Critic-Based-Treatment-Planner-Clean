import numpy as np
import scipy as sp
import glob
import re
import h5py
import h5sparse
from scipy.sparse import csr_matrix, hstack, vstack, csc_matrix
import matplotlib.pyplot as plt
from collections import Counter
import sys

sys.path.append("/home/mainul/Actor-critic-based-treatment-planning/acer_VTPN-12-18-23/Prostate_TORTS/")

import os

datapath1 = 'test_onceagain.hdf5'
datapath2 = 'test_dose_mask_onceagain.h5'
datapath3 = 'test_structure_mask_onceagain.h5'

"""
This is the file to create Dij files in PROSTATE folder

author: Parth Vijaykumar Soni, pxs3648@mavs.uta.edu 
In case if you find any bugs or inconsistency in the code, feel free to contact.
"""

 
try:
    if os.path.exists("test_TORTS.hdf5"):
        os.remove("test_TORTS.hdf5")
except Exception as e:
    print(f"An error occurred: {e}")


#list1 = [0,32, 64, 96, 296, 264, 328]

# iterating over D.mat and BEAMINFO.mat file to extract Dij matrix for each angle and storing them in hdf5 file format
for data_file in glob.glob("/home/mainul/Actor-critic-based-treatment-planning/acer_VTPN-12-18-23/Prostate_TORTS/Prostate_CK_*rCheck.mat"):
    
    pattern = r"Prostate_CK_(\d+)"
    match = re.search(pattern, data_file)
    print(str(data_file))
    
    patient_ind = int(match.group(1))
    patient_ind = "%02d" % patient_ind
    print(patient_ind)
    # with h5py.File(str(data_file), 'r') as file:
    whole_data_file = sp.io.loadmat(data_file)

    # Creating Structure Mask
    try:
        if os.path.exists("test_structure_mask_TORTS.h5"):
            os.remove("test_structure_mask_TORTS.h5")
    except Exception as e:
        print(f"An error occurred: {e}")


    file_name = "test_structure_mask_TORTS{}.h5".format(patient_ind)

    rows, cols, depth = whole_data_file['patient'][0,0]['CT'].shape
    TheThreeStructureIndices = np.zeros(3)
    for i in range(12):
        if whole_data_file['patient'][0,0]['StructureNames'][0,i] == 'PTV 3 mm' or whole_data_file['patient'][0,0]['StructureNames'][0,i] == 'PTV':
            TheThreeStructureIndices[0] = i
            non_necessary, PTV_voxel_num = whole_data_file['patient'][0,0]['SampledVoxels'][0,i].shape
        elif whole_data_file['patient'][0,0]['StructureNames'][0,i] == 'Bladder':
            TheThreeStructureIndices[1] = i
            non_necessary, Bladder_voxel_num = whole_data_file['patient'][0,0]['SampledVoxels'][0,i].shape            
        elif whole_data_file['patient'][0,0]['StructureNames'][0,i] == 'Rectum' or whole_data_file['patient'][0,0]['StructureNames'][0,i] == 'Rectal Wall':
            TheThreeStructureIndices[2] = i
            non_necessary, Rectum_voxel_num = whole_data_file['patient'][0,0]['SampledVoxels'][0,i].shape

    TheThreeStructureIndices = TheThreeStructureIndices.astype(int)
    # The next three lines were the older block
    # non_necessary, PTV_voxel_num = whole_data_file['patient'][0,0]['SampledVoxels'][0,0].shape
    # non_necessary, Bladder_voxel_num = whole_data_file['patient'][0,0]['SampledVoxels'][0,1].shape
    # non_necessary, Rectum_voxel_num = whole_data_file['patient'][0,0]['SampledVoxels'][0,2].shape

    beamlet_num = whole_data_file['data'][0,0]['misc'][0,0]['size'][0][0]
    print('beam', beamlet_num)

    # depth =371
    # cols = 298
    # rows = 407

    total_voxel = depth*rows*cols
    voxel_vector = np.arange(0, total_voxel)
    PTV_indices = np.zeros(PTV_voxel_num).astype(int)
    Bladder_indices = np.zeros(Bladder_voxel_num).astype(int)
    Rectum_indices = np.zeros(Rectum_voxel_num).astype(int)

    print('PTV Voxel Number', PTV_voxel_num)
    print('Rectum Voxel Number', Rectum_voxel_num)
    print('Bladder Voxel Number', Bladder_voxel_num)

    for i in range(PTV_voxel_num):
        x_index = whole_data_file['patient'][0,0]['SampledVoxels'][0,TheThreeStructureIndices[0]][0,i]
        y_index = whole_data_file['patient'][0,0]['SampledVoxels'][0,TheThreeStructureIndices[0]][1,i]
        z_index = whole_data_file['patient'][0,0]['SampledVoxels'][0,TheThreeStructureIndices[0]][2,i]

        PTV_indices[i] = (z_index-1)*(rows*cols) + (x_index-1)*(cols) + y_index

        if i == 3247 or i ==3343:
            print('i', i)
            print('x_index', x_index)
            print('y_index', y_index)
            print('z_index', z_index)
            print('difference', np.shape(np.intersect1d(whole_data_file['data'][0,0]['matrix'][0,0]['A'][3343,:],whole_data_file['data'][0,0]['matrix'][0,0]['A'][3247,:])))
            print('nonZero in 3343', np.shape(np.unique(whole_data_file['data'][0,0]['matrix'][0,0]['A'][i,:])))
    
    result_PTV = np.isin(voxel_vector, PTV_indices).astype(int)

    # =====================================================================Block For getting repeated ones=================
    # Get unique values, inverse indices, and counts
    unique_values, inverse_indices, counts = np.unique(PTV_indices, return_inverse=True, return_counts=True)

    # Find values that appear more than once
    repeated_values = unique_values[counts > 1]

    repeated_indices = {val: np.where(PTV_indices == val)[0] for val in repeated_values}

    #=======================================================================Block for getting Repeated Ones=================

    print('Repeated indices', repeated_indices)
    print('result_PTV',np.shape(np.unique(PTV_indices)))
    print('Not Found ones PTV', PTV_indices[~np.isin(PTV_indices, voxel_vector)])
    # print(result_PTV[99000:100000])

    for i in range(Bladder_voxel_num):
        x_index = whole_data_file['patient'][0,0]['SampledVoxels'][0,TheThreeStructureIndices[1]][0,i]
        y_index = whole_data_file['patient'][0,0]['SampledVoxels'][0,TheThreeStructureIndices[1]][1,i]
        z_index = whole_data_file['patient'][0,0]['SampledVoxels'][0,TheThreeStructureIndices[1]][2,i]

        Bladder_indices[i] = (z_index-1)*(rows*cols) + (x_index-1)*(cols) + y_index
    
    result_Bladder = np.isin(voxel_vector, Bladder_indices).astype(int)
    print('result_Bladder',np.shape(np.nonzero(result_Bladder)))
    print('Not Found ones Bla', Bladder_indices[~np.isin(Bladder_indices, voxel_vector)])
    # print(result_Bladder[99000:100000])

    for i in range(Rectum_voxel_num):
        x_index = whole_data_file['patient'][0,0]['SampledVoxels'][0,TheThreeStructureIndices[2]][0,i]
        y_index = whole_data_file['patient'][0,0]['SampledVoxels'][0,TheThreeStructureIndices[2]][1,i]
        z_index = whole_data_file['patient'][0,0]['SampledVoxels'][0,TheThreeStructureIndices[2]][2,i]

        Rectum_indices[i] = (z_index-1)*(rows*cols) + (x_index-1)*(cols) + y_index
    
    result_Rectum = np.isin(voxel_vector, Rectum_indices).astype(int)
    print('result_Rectum',np.shape(np.nonzero(result_Rectum)))
    print('Not Found ones Rec', Rectum_indices[~np.isin(Rectum_indices, voxel_vector)])

    # The next 26 lines were the previous version
    # for i in range(PTV_voxel_num):
    #     x_index = whole_data_file['patient'][0,0]['SampledVoxels'][0,0][0,i]
    #     y_index = whole_data_file['patient'][0,0]['SampledVoxels'][0,0][1,i]
    #     z_index = whole_data_file['patient'][0,0]['SampledVoxels'][0,0][2,i]

    #     PTV_indices[i] = (z_index-1)*(rows*cols) + (x_index-1)*(cols) + y_index
    
    # result_PTV = np.isin(voxel_vector, PTV_indices).astype(int)
    # # print(result_PTV[99000:100000])

    # for i in range(Bladder_voxel_num):
    #     x_index = whole_data_file['patient'][0,0]['SampledVoxels'][0,1][0,i]
    #     y_index = whole_data_file['patient'][0,0]['SampledVoxels'][0,1][1,i]
    #     z_index = whole_data_file['patient'][0,0]['SampledVoxels'][0,1][2,i]

    #     Bladder_indices[i] = (z_index-1)*(rows*cols) + (x_index-1)*(cols) + y_index
    
    # result_Bladder = np.isin(voxel_vector, Bladder_indices).astype(int)
    # # print(result_Bladder[99000:100000])

    # for i in range(Rectum_voxel_num):
    #     x_index = whole_data_file['patient'][0,0]['SampledVoxels'][0,2][0,i]
    #     y_index = whole_data_file['patient'][0,0]['SampledVoxels'][0,2][1,i]
    #     z_index = whole_data_file['patient'][0,0]['SampledVoxels'][0,2][2,i]

    #     Rectum_indices[i] = (z_index-1)*(rows*cols) + (x_index-1)*(cols) + y_index
    
    # result_Rectum = np.isin(voxel_vector, Rectum_indices).astype(int)


    # print(result_Rectum[99000:100000])
    # print('lastThreeWereResults')
    # # This is the wrong method for Creating list of voxel indices list for entire body
    # PTV_vector_ones = np.ones(4976)
    # PTV_vector_zeroes = np.zeros(4976)
    # Bladder_vector_ones = np.ones(4976)
    # Bladder_vector_zeroes = np.zeros(4976)
    # Rectum_vector_ones = np.ones(4981)
    # Rectum_vector_zeroes = np.zeros(4981)
    # # voxel_vector = np.arange(0, 14933)
    # PTV_indices = np.concatenate((PTV_vector_ones, Bladder_vector_zeroes, Rectum_vector_zeroes))
    # Bladder_indices = np.concatenate((PTV_vector_zeroes, Bladder_vector_ones, Rectum_vector_zeroes))
    # Rectum_indices = np.concatenate((PTV_vector_zeroes, Bladder_vector_zeroes, Rectum_vector_ones))
    with h5py.File(file_name, 'w') as hdf5_file:
        group_name = 'oar_ptvs'
        group = hdf5_file.create_group(group_name)

        group.create_dataset('PTV', data = result_PTV)
        group.create_dataset('Bladder', data = result_Bladder)
        group.create_dataset('Rectum', data = result_Rectum)

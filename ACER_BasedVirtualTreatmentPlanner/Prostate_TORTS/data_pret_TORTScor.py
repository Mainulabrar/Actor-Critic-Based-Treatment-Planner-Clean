import numpy as np
import scipy as sp
import glob
import re
import h5py
import h5sparse
from scipy.sparse import csr_matrix, hstack, vstack, csc_matrix
import matplotlib.pyplot as plt
from collections import Counter


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
for data_file in glob.glob("./Prostate_CK_*r.mat"):
    
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

    for i in range(PTV_voxel_num):
        x_index = whole_data_file['patient'][0,0]['SampledVoxels'][0,TheThreeStructureIndices[0]][0,i]
        y_index = whole_data_file['patient'][0,0]['SampledVoxels'][0,TheThreeStructureIndices[0]][1,i]
        z_index = whole_data_file['patient'][0,0]['SampledVoxels'][0,TheThreeStructureIndices[0]][2,i]

        PTV_indices[i] = (z_index-1)*(rows*cols) + (x_index-1)*(cols) + y_index
    
    result_PTV = np.isin(voxel_vector, PTV_indices).astype(int)
    # print(result_PTV[99000:100000])

    for i in range(Bladder_voxel_num):
        x_index = whole_data_file['patient'][0,0]['SampledVoxels'][0,TheThreeStructureIndices[1]][0,i]
        y_index = whole_data_file['patient'][0,0]['SampledVoxels'][0,TheThreeStructureIndices[1]][1,i]
        z_index = whole_data_file['patient'][0,0]['SampledVoxels'][0,TheThreeStructureIndices[1]][2,i]

        Bladder_indices[i] = (z_index-1)*(rows*cols) + (x_index-1)*(cols) + y_index
    
    result_Bladder = np.isin(voxel_vector, Bladder_indices).astype(int)
    # print(result_Bladder[99000:100000])

    for i in range(Rectum_voxel_num):
        x_index = whole_data_file['patient'][0,0]['SampledVoxels'][0,TheThreeStructureIndices[2]][0,i]
        y_index = whole_data_file['patient'][0,0]['SampledVoxels'][0,TheThreeStructureIndices[2]][1,i]
        z_index = whole_data_file['patient'][0,0]['SampledVoxels'][0,TheThreeStructureIndices[2]][2,i]

        Rectum_indices[i] = (z_index-1)*(rows*cols) + (x_index-1)*(cols) + y_index
    
    result_Rectum = np.isin(voxel_vector, Rectum_indices).astype(int)

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

    # creating Dij

    file_name = "test_TORTS{}.hdf5".format(patient_ind)
    
    # Code for correctly getting the PTV, Bladder and Rectum
    for i in range(12):
        if whole_data_file['data'][0,0]['matrix'][0,i]['Name'] == 'PTV 3 mm' or whole_data_file['data'][0,0]['matrix'][0,i]['Name'] == 'PTV':
            Dij_ptv = whole_data_file['data'][0,0]['matrix'][0,i]['A']
        elif whole_data_file['data'][0,0]['matrix'][0,i]['Name'] == 'Bladder':
            Dij_bladder = whole_data_file['data'][0,0]['matrix'][0,i]['A']            
        elif whole_data_file['data'][0,0]['matrix'][0,i]['Name'] == 'Rectum' or whole_data_file['data'][0,0]['matrix'][0,i]['Name'] == 'Rectal Wall':
            Dij_rectum = whole_data_file['data'][0,0]['matrix'][0,i]['A']


    # The next 4 lines were the previous block
    # Dij_ptv = whole_data_file['data'][0,0]['matrix'][0,0]['A']
    # # print('Dij_ptv', Dij_ptv)
    # Dij_bladder = whole_data_file['data'][0,0]['matrix'][0,4]['A']
    # Dij_rectum = whole_data_file['data'][0,0]['matrix'][0,2]['A']


    Dij = vstack([Dij_ptv, Dij_bladder, Dij_rectum])
    # print('Dij', Dij)

    # My trial to get the correct Dij
    # Dij_nonzeoro_ = np.nonzeros(Dij_ptv)

    row_indices_original = np.concatenate((PTV_indices, Bladder_indices, Rectum_indices))
    row_indices, column_indices = np.nonzero(Dij)
    # row_indices_update = np.zeros(row_indices.shape)
    # for i, x in enumerate(row_indices):
    #     row_indices_update[i] = row_indices_original[x]

    # Get unique elements and their counts
    effective_loop_run = 0
    # effective_voxel_num
    index_to_delete = np.zeros(row_indices.shape)
    Check_array = np.zeros(row_indices.shape)
    for i in range(len(row_indices_original)):
        if row_indices_original[i] in Check_array:
            index_to_delete[i- effective_loop_run] = i
        else: 
            Check_array[effective_loop_run] = row_indices_original[i]
            effective_loop_run +=1

    row_indices_update1 = Check_array[0:(effective_loop_run-1)]
    index_to_delete = index_to_delete[0: (len(row_indices_original)- effective_loop_run)]
    index_to_delete = index_to_delete.astype(int)

    Dij = Dij.toarray()

    Dij = np.delete(Dij, index_to_delete, axis = 0)

    row_indices, column_indices = np.nonzero(Dij)
    row_indices_original1 = np.delete(row_indices_original, index_to_delete, axis = 0)
    print('row_indices_original1.shape', row_indices_original1.shape)
    row_indices_update = np.zeros(row_indices.shape)
    print('row_indices_update.shape', row_indices_update.shape)
    print('row_indices.max', max(row_indices))
    for i, x in enumerate(row_indices):
        row_indices_update[i] = row_indices_original1[x]

    print('index_to_delete', index_to_delete)
    print('Check_array_shape',Check_array.shape)
    print('Dij_shape',Dij.shape)


    # unique_row_elements, counts = np.unique(row_indices_original, return_counts=True)

    # #     # Count occurrences of each element in the array
    # # element_counts = Counter(row_indices_original)

    # # # Filter elements with counts greater than 1 (repeated elements)
    # # repeated_elements = {element: count for element, count in element_counts.items() if count > 1}

    # print("unique_row_elements elements and their counts:", unique_row_elements, counts)

    # # Find repeated elements
    # repeated_row_elements = unique_row_elements[counts > 1]
    # print('repeated_row_elements', repeated_row_elements.shape)




    # column_indices = Dij.nonzero()[0]

    print('column indices', column_indices)
    print('column indices_shape', column_indices.shape)

    Dij_transpose = Dij.transpose()
    # print('Dij_transpose', Dij_transpose)

    Dij_transpose_sparse = csc_matrix(Dij_transpose)
    data_transpose = Dij_transpose_sparse.data
    print('data_transpose', data_transpose.shape)
    rows_transpose = column_indices
    column_transpose = row_indices_update
    print('column_transpose', column_transpose.shape)
    print('rows_transpose', rows_transpose.shape)

    # print('got_this_far', data_transpose.shape, rows_transpose.shape, column_transpose.shape) 
    Dij_new = csr_matrix((data_transpose, (rows_transpose, column_transpose)), shape = (beamlet_num, total_voxel))
    print('Dij_transpose.shape',Dij_transpose.shape)
    # print(Dij_ptv.type)

    # Oldschool Effort 
    # Dij_transpose = Dij.transpose()
    # Dij_new = csr_matrix(Dij_transpose)  

    # To Look at Data
    # ToLookAtData = Dij_new.tocsc()
    # print('to look at', ToLookAtData.getcol(column_transpose[0]), ToLookAtData.indices)

    # Extracting the data, indices and indptr from Dij sparse matrix which is in the csc form.
    data = Dij_new.data
    indices = Dij_new.indices
    indptr = Dij_new.indptr

    # Getting beamlet number form BEAMINFO.mat file
    # beamlet_num = information['numBeamlets'][0][0]
    # subgroup = group.create_group(subgroup_name)

    # Creating dataset in subgroup 'Dij' for all the angles which are there in PROSTATE folder, and using appropriate parameters to compress it into small size
    with h5py.File(file_name, 'w') as hdf5_file:
        group_name = 'Dij'
        group = hdf5_file.create_group(group_name)

        group.create_dataset('data',  data = data, maxshape = data.shape, compression = 'gzip', compression_opts = 9, dtype = h5py.h5t.IEEE_F32LE)
        group.create_dataset('indices', data = indices, maxshape = indices.shape, compression = 'gzip', compression_opts = 9, dtype=h5py.h5t.STD_I32LE)
        group.create_dataset('indptr', data = indptr, maxshape = indptr.shape, compression = 'gzip', compression_opts = 9, dtype=h5py.h5t.STD_I64LE )
        
        # Creating attributes to get extract the information about the Dij matrix during run time
        group.attrs.create("h5sparse_shape", data = Dij_new.shape)
        group.attrs.create("h5sparse_format", data = "csr")

   

    

# Creating Dose Mask
    try:
        if os.path.exists("test_dose_mask_TORTS.h5"):
            os.remove("test_dose_mask_TORTS.h5")
    except Exception as e:
        print(f"An error occurred: {e}")

    final_mask = []
    # voxel_vector = np.arange(0,14933)
    #arr = sp.sparse.csc_array((data, indices, indptr), shape = (14933, 2260))
    non_zero_idx = Dij.nonzero()[0]
    print(non_zero_idx.shape)

    #finding the unique number of voxel which are affected by the beamlet
    idx, counts = np.unique(non_zero_idx, return_counts=True)
    print('idx shape', counts, idx.shape)


    final_mask_no_overlap = np.union1d(PTV_indices, np.union1d(Bladder_indices, Rectum_indices))
    final_mask_overlap = np.concatenate((PTV_indices, Bladder_indices, Rectum_indices))
    print("final_mask_no_overlap", final_mask_no_overlap.shape[0])
    print("final_mask_overlap", final_mask_overlap.shape[0])
    result = np.isin(voxel_vector, final_mask_no_overlap).astype(int)
    # print("np.nonzero(result)",np.nonzero(result))
    
    file_name = "test_dose_mask_TORTS{}.h5".format(patient_ind)
    with h5py.File(file_name, 'w') as hdf5_file:
        hdf5_file.create_dataset('dose', data = result)

'''
    # printing the dose colorwash
    Dose_matrix = np.ones(beamlet_num)
    Dose_sparse = csr_matrix(Dose_matrix)

    print('Dij_new_shape', Dij_new.shape)
    dose_total = Dose_sparse.dot(Dij_new)
    dose_total1 = dose_total.toarray()
    dose_total2 = dose_total1.reshape(-1)
    print('dose_total', dose_total2)
    # dose_total = dose_total.transpose()
    print('dose_total_size', dose_total2.shape)
    if depth * rows * cols != dose_total2.size:
        raise ValueError("Total number of elements in Dose_3d does not match the product of depth, rows, and cols")
    Dose_3d = dose_total2.reshape((depth, rows, cols))
    print('result_PTV_size', result_PTV.size)

    DosePTV = np.zeros(PTV_indices.size)
    for i, x in enumerate(PTV_indices):
        DosePTV[i] = dose_total2[x] 
        # print('DosePTVElement',DosePTV[i])
    print('DosePTV_nonzero_shape', DosePTV.nonzero()[0].shape)

    DoseBLA = np.zeros(Bladder_indices.size)
    for i, x in enumerate(Bladder_indices):
        DoseBLA[i] = dose_total2[x]
    print('DoseBLA_nonzero_shape', DoseBLA.nonzero()[0].shape)

    DoseRec = np.zeros(Rectum_indices.size)
    for i, x in enumerate(Rectum_indices):
        DoseRec[i] = dose_total2[x]  
    print('DoseRec_nonzero_shape', DoseRec.nonzero()[0].shape)

    PTV_3d = result_PTV.reshape((depth, rows, cols))
    PTV_3d = np.transpose(PTV_3d, (0,2,1))
    Dose_3d = np.transpose(Dose_3d,(0,2,1))
    print('nonzero', PTV_3d.nonzero()[0])
    print()
    Dose_2d = Dose_3d[110]
    PTV_2d = PTV_3d[110]
    
    plt.imshow(Dose_2d, cmap = 'jet', interpolation ='nearest')
    plt.colorbar()
    plt.show()


    # For plotting DVH against changing and fixed both edge_ptv
    edge_ptv = np.zeros((100 + 1,))
    edge_ptv_max = np.zeros((100 + 1,))                        
    edge_ptv_max[1:100 + 1] = np.linspace(0, max(DosePTV), 100)
    print('maxDosePTV', max(DosePTV), len)
    x_ptv = np.linspace(0.5 * max(DosePTV) / 100, max(DosePTV), 100)
    (n_ptv, b) = np.histogram(DosePTV, bins=edge_ptv)
    (n_ptv_max, b_max) = np.histogram(DosePTV, bins=edge_ptv_max)
    y_ptv = 1 - np.cumsum(n_ptv / len(DosePTV), axis=0)
    y_ptv_max = 1 - np.cumsum(n_ptv_max / len(DosePTV), axis=0)

    plt.plot(edge_ptv_max[1:100+1], y_ptv_max, color = 'red', label = 'PTV')
    # plt.legend(fontsize='5')
    # plt.show()

    edge_bladder = np.zeros((100 + 1,))
    edge_bladder_max = np.zeros((100 + 1,))
    # edge_bladder[1:100 + 1] = np.linspace(0, pdose*1.15, 100)
    edge_bladder_max[1:100 + 1] = np.linspace(0, max(DoseBLA), 100)                            
    x_bladder = np.linspace(0.5 * max(DoseBLA) / 100, max(DoseBLA), 100)
    (n_bladder, b) = np.histogram(DoseBLA, bins=edge_bladder)
    (n_bladder_max, b_max) = np.histogram(DoseBLA, bins = edge_bladder_max)
    y_bladder = 1 - np.cumsum(n_bladder / len(DoseBLA), axis=0)
    y_bladder_max = 1 - np.cumsum(n_bladder_max/len(DoseBLA), axis = 0)

    plt.plot(edge_bladder_max[1:100+1], y_bladder_max, color = 'green', label = 'Bladder')
    # plt.legend(fontsize='5')
    # plt.show()

    edge_rectum = np.zeros((100 + 1,))
    edge_rectum_max = np.zeros((100 + 1,))
    # edge_rectum[1:100 + 1] = np.linspace(0, pdose*1.15, 100)
    edge_rectum_max[1:100 + 1] = np.linspace(0, max(DoseRec), 100)                            
    x_rectum = np.linspace(0.5 * max(DoseRec) / 100, max(DoseRec), 100)
    (n_rectum, b) = np.histogram(DoseRec, bins=edge_rectum)
    (n_rectum_max, b_max) = np.histogram(DoseRec , bins = edge_rectum_max)
    y_rectum = 1 - np.cumsum(n_rectum / len(DoseRec), axis=0)
    y_rectum_max = 1 - np.cumsum(n_rectum_max / len(DoseRec), axis = 0)

    plt.plot(edge_rectum_max[1:100+1], y_rectum_max, color = 'blue', label = 'Rectum')
    plt.legend(fontsize='5')
    plt.show()

    # Y = np.zeros((100, 12))
    # Y[:, 0] = y_ptv
    # Y[:, 1] = y_bladder
    # Y[:, 2] = y_rectum

    # # X = np.zeros((1000, 3))
    # Y[:, 3] = x_ptv
    # Y[:, 4] = x_bladder
    # Y[:, 5] = x_rectum

    # # storing max range histograms
    # Y[:, 6] = y_ptv_max
    # Y[:, 7] = y_bladder_max
    # Y[:, 8] = y_rectum_max

    # Y[:, 9] = edge_ptv_max[1:100+1]
    # Y[:, 10] = edge_bladder_max[1:100+1]
    # Y[:, 11] = edge_rectum_max[1:100+1]

    # Volume_axis = []
    # B_Volume_axis = []
    # R_Volume_axis = []
    # dose_axis_PTV = []
    # dose_axis_BLA = []
    # dose_axis_REC = []

    # colors = plt.cm.rainbow(np.linspace(0, 1, 3))




def loadDoseMatrix(filename):
    test = h5sparse.File(filename,'r')
    Dmat = test['Dij'].value
    Dmat = Dmat.transpose()
    # print("Dmat", Dmat)
    # print("Dmat.shape", Dmat.shape)
    # print("np.nonzero(Dmat)",np.nonzero(Dmat))
    print('Dmat_shape', Dmat.shape)
    return Dmat

variable = loadDoseMatrix('test_TORTS.hdf5')
print(variable.shape)
        
def loadMask(filename1, filename2):
    file1 = h5py.File(filename1,'r')
    file2 = h5py.File(filename2,'r')
    dosemask = file1['dose']
    dose_ind = np.nonzero(dosemask[:])
    # print("dosemask[:]",dosemask[:])
    # print("dose_ind",dose_ind)
    
    PTVtemp = file2['oar_ptvs']['PTV']
    PTV_ind = np.nonzero(PTVtemp)
    PTV_val = np.intersect1d(dose_ind,PTV_ind)
 

    
    bladdertemp = file2['oar_ptvs']['Bladder']
    blad_ind = np.nonzero(bladdertemp)
    blad_val = np.intersect1d(dose_ind,blad_ind)
    # bladder  = bladdertemp[:][dose_ind]

    
    rectumtemp = file2['oar_ptvs']['Rectum']
    rec_ind = np.nonzero(rectumtemp)
    # print("rec_ind",rec_ind)
    rec_val = np.intersect1d(dose_ind,rec_ind)
    # print("rec_val",rec_val)
    # rectum = rectumtemp[:][dose_ind]
    # print("rectum.shape",rectum)
  
    # rectum = rectumtemp[np.nonzero(dosemask)]
    
    targetLabelFinal = np.zeros((dosemask.shape))
    targetLabelFinal[blad_val] = 2
    targetLabelFinal[rec_val] = 3
    targetLabelFinal[PTV_val] = 1
    # print("targetLabelFinal",targetLabelFinal)
    
    bladderLabel = np.zeros((dosemask.shape))
    bladderLabel[blad_val] = 1
    
    rectumLabel = np.zeros((dosemask.shape))
    # rectumLabel1 = np.zeros((dosemask.shape))
    # rectumLabel[np.nonzero(rectumtemp)] = 1
    rectumLabel[rec_val] = 1
    # MRECtemp1 = doseMatrix[rectumLabel == 1, :]
    # print("MRECtemp1",MRECtemp1)
    # MRECtemp2 = doseMatrix[rectumLabel1 == 1, :]
    # print("MRECtemp2",MRECtemp2)
    
    PTVLabel = np.zeros((dosemask.shape))
    PTVLabel[PTV_val] = 1
    return targetLabelFinal, bladderLabel, rectumLabel, PTVLabel


variable1, variable2, variable3, variable4 = loadMask('test_dose_mask_TORTS.h5', 'test_structure_mask_TORTS.h5')
print(variable1.shape, variable2.shape, variable3.shape, variable4.shape)

def ProcessDmat(doseMatrix, targetLabels, bladderLabel, rectumLabel):
    x = np.ones((doseMatrix.shape[1],))
    # print("targetLabelFinal",targetLabelFinal)
    # print("doseMatrix[480136, :]",doseMatrix[480136, :])
    
    
    
    MPTVtemp = doseMatrix[targetLabels == 1, :]
    # print("MPTVtemp",MPTVtemp)
    print("MPTVtemp.shape",MPTVtemp.shape)
    # print("x.shape",x.shape)
    DPTV = MPTVtemp.dot(x)
    MPTV = MPTVtemp[DPTV != 0,:]
    # print("MPTV",MPTV.shape)
    MBLAtemp = doseMatrix[targetLabels == 2, :]
    # print("MBLAtemp",MBLAtemp.shape)
    DBLA = MBLAtemp.dot(x)
    MBLA = MBLAtemp[DBLA != 0,:]
    # print("MBLA",MBLA.shape)
    MRECtemp = doseMatrix[targetLabels == 3, :]
    # print("MRECtemp.shape",MRECtemp.shape)
    DREC = MRECtemp.dot(x)
    MREC = MRECtemp[DREC != 0,:]
    # print("MREC",MREC.shape)
    

    MBLAtemp1 = doseMatrix[bladderLabel == 1, :]
    DBLA1 = MBLAtemp1.dot(x)
    MBLA1 = MBLAtemp1[DBLA1 != 0,:]
    # print("MBLA1",MBLA1.shape)
    MRECtemp1 = doseMatrix[rectumLabel == 1, :]
    # print("MRECtemp1",MRECtemp1)
    DREC1 = MRECtemp1.dot(x)
    MREC1 = MRECtemp1[DREC1 != 0,:]
    # print("MREC1",MREC1)
    # print("MREC1",MREC1.shape)
    return MPTV, MBLA, MREC, MBLA1, MREC1
        





# ========================
# This is correct one.
# targetLabelFinal = np.zeros((dosemask.shape))


# ========================
# Intersection1d makes sense betn dose_ind and rec_ind
# dose_ind (array([ 480136,  499820,  500192, ..., 2972050, 2972235, 2972239]),)
# PTVtemp <HDF5 dataset "PTV_68": shape (3047040,), type "<i8">
# PTVtemp[:] [0 0 0 ... 0 0 0]
# PTVtemp[:][2] 0
# np.nonzero(PTVtemp[:]) (array([1606411, 1606412, 1606413, ..., 2352709, 2352891, 2352892]),)
# np.nonzero(PTV) (array([202565, 202566, 202567, ..., 371023, 371092, 371093]),)
# rec_ind (array([1301524, 1301525, 1301526, ..., 2589347, 2589348, 2589349]),)
# rec_val [1301524 1301525 1301526 ... 2589347 2589348 2589349]

        

'''      


        





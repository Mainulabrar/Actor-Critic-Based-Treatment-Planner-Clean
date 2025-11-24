import numpy as np
import scipy as sp
import glob
import re
import h5py
import h5sparse
from scipy.sparse import csr_matrix, hstack, vstack
import matplotlib.pyplot as plt

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
for data_file in glob.glob("./Prostate_CK_*r3.mat"):
    
    final_mask = []
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
    # hdf5_file = h5py.File(file_name, 'a')
    # group_name = 'oar_ptvs'
    # group = hdf5_file.create_group(group_name)

    rows, cols, depth = whole_data_file['patient'][0,0]['CT'].shape

    # getting the voxel numbers for PTV, Bladder and Rectum
    for i in range(12):
        if whole_data_file['patient'][0,0]['StructureNames'][0,i] == 'PTV 3 mm':
            non_necessary, PTV_voxel_num = whole_data_file['patient'][0,0]['SampledVoxels'][0,i].shape
        elif whole_data_file['patient'][0,0]['StructureNames'][0,i] == 'Bladder':
            non_necessary, Bladder_voxel_num = whole_data_file['patient'][0,0]['SampledVoxels'][0,i].shape            
        elif whole_data_file['patient'][0,0]['StructureNames'][0,i] == 'Rectum' or whole_data_file['patient'][0,0]['StructureNames'][0,i] == 'Rectal Wall':
            non_necessary, Rectum_voxel_num = whole_data_file['patient'][0,0]['SampledVoxels'][0,2].shape
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
    voxel_vector = np.arange(0, (PTV_voxel_num+Bladder_voxel_num+Rectum_voxel_num))

    # PTV_indices = np.arange(PTV_voxel_num).astype(int)
    # Bladder_indices = np.arange(PTV_voxel_num, (PTV_voxel_num+Bladder_voxel_num)).astype(int)
    # Rectum_indices = np.arange((PTV_voxel_num+Bladder_voxel_num), (PTV_voxel_num+Bladder_voxel_num+Rectum_voxel_num)).astype(int)
    # PTV_indices = np.zeros(PTV_voxel_num).astype(int)
    # Bladder_indices = np.zeros(PTV_voxel_num, (PTV_voxel_num+Bladder_voxel_num)).astype(int)
    # Rectum_indices = np.zeros((PTV_voxel_num+Bladder_voxel_num), (PTV_voxel_num+Bladder_voxel_num+Rectum_voxel_num)).astype(int)

    # print(result_Rectum[99000:100000])
    # print('lastThreeWereResults')
    # # This is the wrong method for Creating list of voxel indices list for entire body
    PTV_vector_ones = np.ones(PTV_voxel_num)
    PTV_vector_zeroes = np.zeros(PTV_voxel_num)
    Bladder_vector_ones = np.ones(Bladder_voxel_num)
    Bladder_vector_zeroes = np.zeros(Bladder_voxel_num)
    Rectum_vector_ones = np.ones(Rectum_voxel_num)
    Rectum_vector_zeroes = np.zeros(Rectum_voxel_num)
    # voxel_vector = np.arange(0, 14933)
    PTV_indices = np.concatenate((PTV_vector_ones, Bladder_vector_zeroes, Rectum_vector_zeroes)).astype(int)
    Bladder_indices = np.concatenate((PTV_vector_zeroes, Bladder_vector_ones, Rectum_vector_zeroes)).astype(int)
    Rectum_indices = np.concatenate((PTV_vector_zeroes, Bladder_vector_zeroes, Rectum_vector_ones)).astype(int)
    with h5py.File(file_name, 'w') as hdf5_file:
        group_name = 'oar_ptvs'
        group = hdf5_file.create_group(group_name)

        group.create_dataset('PTV', data = PTV_indices)
        group.create_dataset('Bladder', data = Bladder_indices)
        group.create_dataset('Rectum', data = Rectum_indices)

    # group.create_dataset('PTV', data = PTV_indices)
    # group.create_dataset('Bladder', data = Bladder_indices)
    # group.create_dataset('Rectum', data = Rectum_indices)

    # creating Dij

    file_name = "test_TORTS{}.hdf5".format(patient_ind)
    # hdf5_file = h5py.File(file_name, 'a')
    # group_name = 'Dij'
    # group = hdf5_file.create_group(group_name)


    Dij_ptv = whole_data_file['data'][0,0]['matrix'][0,0]['A']
    # print('Dij_ptv', Dij_ptv)
    Dij_bladder = whole_data_file['data'][0,0]['matrix'][0,4]['A']
    Dij_rectum = whole_data_file['data'][0,0]['matrix'][0,2]['A']


    Dij = vstack([Dij_ptv, Dij_bladder, Dij_rectum])
    # print('Dij', Dij)

    # My trial to get the correct Dij
    # Dij_nonzeoro_ = np.nonzeros(Dij_ptv)

    Dij_transpose = Dij.transpose()

    Dij_new = csr_matrix(Dij_transpose)
    print('Dij_transpose.shape',Dij_transpose.shape)
    # print(Dij_ptv.type)

    # Oldschool Effort 
    # Dij_transpose = Dij.transpose()
    # Dij_new = csr_matrix(Dij_transpose)  

    # Extracting the data, indices and indptr from Dij sparse matrix which is in the csc form.
    data = Dij_new.data
    indices = Dij_new.indices
    indptr = Dij_new.indptr

    # Getting beamlet number form BEAMINFO.mat file
    # beamlet_num = information['numBeamlets'][0][0]
    # subgroup = group.create_group(subgroup_name)
    with h5py.File(file_name, 'w') as hdf5_file:
        group_name = 'Dij'
        group = hdf5_file.create_group(group_name)

        group.create_dataset('data',  data = data, maxshape = data.shape, compression = 'gzip', compression_opts = 9, dtype = h5py.h5t.IEEE_F32LE)
        group.create_dataset('indices', data = indices, maxshape = indices.shape, compression = 'gzip', compression_opts = 9, dtype=h5py.h5t.STD_I32LE)
        group.create_dataset('indptr', data = indptr, maxshape = indptr.shape, compression = 'gzip', compression_opts = 9, dtype=h5py.h5t.STD_I64LE )
        
        # Creating attributes to get extract the information about the Dij matrix during run time
        group.attrs.create("h5sparse_shape", data = Dij_new.shape)
        group.attrs.create("h5sparse_format", data = "csr")

    # # Creating dataset in subgroup 'Dij' for all the angles which are there in PROSTATE folder, and using appropriate parameters to compress it into small size
    # group.create_dataset('data',  data = data, maxshape = data.shape, compression = 'gzip', compression_opts = 9, dtype = h5py.h5t.IEEE_F32LE)
    # group.create_dataset('indices', data = indices, maxshape = indices.shape, compression = 'gzip', compression_opts = 9, dtype=h5py.h5t.STD_I32LE)
    # group.create_dataset('indptr', data = indptr, maxshape = indptr.shape, compression = 'gzip', compression_opts = 9, dtype=h5py.h5t.STD_I64LE )
    
    # # Creating attributes to get extract the information about the Dij matrix during run time
    # group.attrs.create("h5sparse_shape", data = Dij_new.shape)
    # group.attrs.create("h5sparse_format", data = "csr")

   

    

# Creating Dose Mask
    try:
        if os.path.exists("test_dose_mask_TORTS.h5"):
            os.remove("test_dose_mask_TORTS.h5")
    except Exception as e:
        print(f"An error occurred: {e}")


    # voxel_vector = np.arange(0,14933)
    #arr = sp.sparse.csc_array((data, indices, indptr), shape = (14933, 2260))
    non_zero_idx = Dij.nonzero()[0]
    print(non_zero_idx.shape)

    #finding the unique number of voxel which are affected by the beamlet
    idx, counts = np.unique(non_zero_idx, return_counts=True)
    print('idx shape', counts, idx.shape)


    final_mask = np.union1d(final_mask, idx)

    result = np.isin(voxel_vector, final_mask).astype(int)
    # print("np.nonzero(result)",np.nonzero(result))
    print('dose_mask_shape', result.size)
    
    file_name = "test_dose_mask_TORTS{}.h5".format(patient_ind)
    with h5py.File(file_name, 'w') as hdf5_file:
        hdf5_file.create_dataset('dose', data = result)


    # creating DVH
    Dose_matrix = np.ones(beamlet_num)
    Dose_sparse = csr_matrix(Dose_matrix)

    print('Dij_new_shape', Dij_new.shape)
    dose_total = Dose_sparse.dot(Dij_new)
    dose_total1 = dose_total.toarray()
    dose_total2 = dose_total1.reshape(-1)
    print('dose_total', dose_total2)
    # dose_total = dose_total.transpose()
    print('dose_total_size', dose_total2.shape)

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

'''
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

'''
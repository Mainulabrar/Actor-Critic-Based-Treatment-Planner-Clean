import numpy as np
import scipy as sp
import glob
import re
import h5py
from scipy.sparse import csr_matrix, hstack

import os

datapath1 = 'test_onceagain.hdf5'
datapath2 = 'test_dose_mask_onceagain.h5'
datapath3 = 'test_structure_mask_onceagain.h5'
 
try:
    if os.path.exists("test_onceagain.hdf5"):
        os.remove("test_onceagain.hdf5")
except Exception as e:
    print(f"An error occurred: {e}")

file_name = "test_onceagain.hdf5"
hdf5_file = h5py.File(file_name, 'a')
group_name = 'Dij'
group = hdf5_file.create_group(group_name)
list1 = [0,32, 64, 96, 296, 264, 328]

# iterating over D.mat and BEAMINFO.mat file to extract Dij matrix for each angle and storing them in hdf5 file format
for data_file, inf_file in zip(glob.glob("/home/mainul/Actor-critic-based-treatment-planning/acer_VTPN-12-18-23/PROSTATE/Gantry*_Couch*_D.mat"), glob.glob("/home/mainul/Actor-critic-based-treatment-planning/acer_VTPN-12-18-23/PROSTATE/Gantry*_Couch*_BEAMINFO.mat")):
    pattern = r"Gantry(\d+)"
    match = re.search(pattern, data_file)
    
    gantry_angle = int(match.group(1))
    
    
    if match and (gantry_angle in list1):
        print("gantry_angle",gantry_angle)
        Dij = sp.io.loadmat(data_file)
        information = sp.io.loadmat(inf_file)
        gantry_angle = int(match.group(1))
        subgroup_name = f'{gantry_angle:03}'
        Dij_normal = Dij['D'].transpose()
        Dij_new = csr_matrix(Dij_normal)
        print('Dij_new', Dij_new.shape)

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
        
try:
    if os.path.exists('test_dose_mask_onceagain.h5'):
        os.remove('test_dose_mask_onceagain.h5')
except Exception as e:
    print(f"An error occurred: {e}")

final_mask = []
depth = 90 
rows = 184
cols = 184
total_voxel = depth*rows*cols
voxel_vector = np.arange(1, total_voxel + 1)

# Loop through all the angles to find the idices of voxel affected in each angle
for i in range(len(list1)):

    Dij = sp.io.loadmat(f"/home/mainul/Actor-critic-based-treatment-planning/acer_VTPN-12-18-23/PROSTATE/Gantry{list1[i]}_Couch0_D.mat")
    information = sp.io.loadmat(f"/home/mainul/Actor-critic-based-treatment-planning/acer_VTPN-12-18-23/PROSTATE/Gantry{list1[i]}_Couch0_BEAMINFO.mat")

    data = Dij['D'].data
    indices = Dij['D'].indices
    # idx1 = [x + 1 for x in indices]
    indptr = Dij['D'].indptr
    
    # finding out the number of beamlet, because number for beamlet for every angle is different
    num_beamlets = information['numBeamlets'][0][0]
    arr = sp.sparse.csc_array((data, indices, indptr), shape = (3047040,num_beamlets))

    #getting the non-zero indices
    non_zero_idx = arr.nonzero()[0]

    #finding the unique number of voxel which are affected by the beamlet
    idx, counts = np.unique(non_zero_idx, return_counts=True)
    idx1 = [x + 1 for x in idx]

    #union with the final_mask, final_mask will contain the unique indices with has all the affected voxel in each angle, this will be updated in each iteration 
    final_mask = np.union1d(final_mask, idx1)

    # print(f"No of non-zero index in final_mask after {i} th Gantry angle is {np.sum(np.isin(voxel_vector, final_mask))}")


print("No of voxel affected by beamlet", final_mask.shape[0])

#converting the final_mask into binary vector
result = np.isin(voxel_vector, final_mask).astype(int)
print("np.nonzero(result)",np.nonzero(result))

file_name = 'test_dose_mask_onceagain.h5'
hdf5_file = h5py.File(file_name, 'a')
hdf5_file.create_dataset('dose', data = result)

try:
    if os.path.exists("test_structure_mask_onceagain.h5"):
        os.remove("test_structure_mask_onceagain.h5")
except Exception as e:
    print(f"An error occurred: {e}")


file_name = "test_structure_mask_onceagain.h5"
hdf5_file = h5py.File(file_name, 'a')
group_name = 'oar_ptvs'
group = hdf5_file.create_group(group_name)

depth = 90 
rows = 184
cols = 184

total_voxel = depth*rows*cols

# Created list of voxel indices list for entire body
voxel_vector = np.arange(1, total_voxel + 1)

# Iterating over entire VOLIST for Organs at Risk (OARs) and Planning Target Volume (PTVs)
for fname in glob.glob("/home/mainul/Actor-critic-based-treatment-planning/acer_VTPN-12-18-23/PROSTATE/*_VOILIST.mat"): 

    # Used regex to find the organ or ptv name
    pattern = r'([^/]+)\.mat$'
    match = re.search(pattern, fname)
    if match:
        mat_string = match.group()
        extracted_string = mat_string.split('_VOILIST')[0]
        # print("extracted_string",extracted_string)
        structure_mask = sp.io.loadmat(fname)
        structure_mask_flat = structure_mask['v'].flatten()
        intersection = np.intersect1d(voxel_vector, structure_mask_flat)

        #creating binary vector
        result = np.isin(voxel_vector, intersection).astype(int)
        print(f"{extracted_string}",np.sum(result), len(result))

        group.create_dataset(f"{extracted_string}", data = result)


def loadDoseMatrix(filename):
    test = h5sparse.File(filename,'r')
    Dmat = test['Dij']['000'].value
    temp = test['Dij']['032'].value
    Dmat = vstack([Dmat, temp])
    temp = test['Dij']['064'].value
    Dmat = vstack([Dmat, temp])
    temp = test['Dij']['096'].value
    Dmat = vstack([Dmat, temp])
    temp = test['Dij']['296'].value
    Dmat = vstack([Dmat, temp])
    temp = test['Dij']['264'].value
    Dmat = vstack([Dmat, temp])
    temp = test['Dij']['328'].value
    Dmat = vstack([Dmat, temp])
    Dmat = Dmat.transpose()
    return Dmat

def loadMask(filename1, filename2):
    file1 = h5py.File(filename1,'r')
    file2 = h5py.File(filename2,'r')
    dosemask = file1['dose']
    dose_ind = np.nonzero(dosemask[:])
    
    PTVtemp = file2['oar_ptvs']['PTV_68']
    PTV_ind = np.nonzero(PTVtemp)
    PTV_val = np.intersect1d(dose_ind,PTV_ind)
 
    bladdertemp = file2['oar_ptvs']['Bladder']
    blad_ind = np.nonzero(bladdertemp)
    blad_val = np.intersect1d(dose_ind,blad_ind)
    
    rectumtemp = file2['oar_ptvs']['Rectum']
    rec_ind = np.nonzero(rectumtemp)
    rec_val = np.intersect1d(dose_ind,rec_ind)
    
    targetLabelFinal = np.zeros((dosemask.shape))
    targetLabelFinal[blad_val] = 2
    targetLabelFinal[rec_val] = 3
    targetLabelFinal[PTV_val] = 1
    
    bladderLabel = np.zeros((dosemask.shape))
    bladderLabel[blad_val] = 1
    
    rectumLabel = np.zeros((dosemask.shape))
    rectumLabel[rec_val] = 1
    
    PTVLabel = np.zeros((dosemask.shape))
    PTVLabel[PTV_val] = 1
    return targetLabelFinal, bladderLabel, rectumLabel, PTVLabel


def ProcessDmat(doseMatrix, targetLabels, bladderLabel, rectumLabel):
    x = np.ones((doseMatrix.shape[1],))
      
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
        

        


        





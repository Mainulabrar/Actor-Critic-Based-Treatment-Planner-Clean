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

def loadDoseMatrix(filename):
    test = h5sparse.File(filename,'r')
    Dmat = test['Dij'].value
    Dmat = Dmat.transpose()
    # print("Dmat", Dmat)
    # print("Dmat.shape", Dmat.shape)
    # print("np.nonzero(Dmat)",np.nonzero(Dmat))
    print('Dmat_shape', Dmat.shape)
    return Dmat

# variable = loadDoseMatrix('test_TORTS.hdf5')
# print(variable.shape)
        
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


# variable1, variable2, variable3, variable4 = loadMask('test_dose_mask_TORTS.h5', 'test_structure_mask_TORTS.h5')
# print(variable1.shape, variable2.shape, variable3.shape, variable4.shape)

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

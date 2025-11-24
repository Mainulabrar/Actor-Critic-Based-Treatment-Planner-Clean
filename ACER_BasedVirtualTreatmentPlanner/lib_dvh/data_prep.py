# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 10:48:28 2019

@author: S191914
"""
import h5sparse
import h5py
import numpy as np
from scipy.sparse import vstack

data_path = '/media/mainul/Chi-Drives/128641-3-E21/var/lib/docker/overlay2/8fa0092f3bc6971c1752f734f8fd34c782377f383ffe6f5a55629a01d3b0f185/diff/home/exx/dose_deposition_full/prostate_dijs/f_dijs/008.hdf5'
data_path2 = '/media/mainul/Chi-Drives/128641-3-E21/var/lib/docker/overlay2/8fa0092f3bc6971c1752f734f8fd34c782377f383ffe6f5a55629a01d3b0f185/diff/home/exx/dose_deposition_full/plostate_dijs/f_masks/008.h5'



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
    print('DmatShape',Dmat.shape)
    return Dmat

def loadMask(filename):
    mask = h5py.File(filename,'r')
    dosemask = mask['oar_ptvs']['dose']
    print(dosemask.shape)
    dosemask = np.reshape(dosemask, (dosemask.shape[0] * dosemask.shape[1] * dosemask.shape[2],), order='C')
    PTVtemp = mask['oar_ptvs']['ptv']
    PTVtemp = np.reshape(PTVtemp, (PTVtemp.shape[0] * PTVtemp.shape[1] * PTVtemp.shape[2],), order='C')
    PTV = PTVtemp[np.nonzero(dosemask)]
    bladdertemp = mask['oar_ptvs']['bladder']
    bladdertemp = np.reshape(bladdertemp, (bladdertemp.shape[0] * bladdertemp.shape[1] * bladdertemp.shape[2],), order='C')
    bladder = bladdertemp[np.nonzero(dosemask)]
    rectumtemp = mask['oar_ptvs']['rectum']
    rectumtemp = np.reshape(rectumtemp, (rectumtemp.shape[0] * rectumtemp.shape[1] * rectumtemp.shape[2],), order='C')
    rectum = rectumtemp[np.nonzero(dosemask)]
    targetLabelFinal = np.zeros((PTV.shape))
    targetLabelFinal[np.nonzero(bladder)] = 2
    targetLabelFinal[np.nonzero(rectum)] = 3
    targetLabelFinal[np.nonzero(PTV)] = 1
    bladderLabel = np.zeros((PTV.shape))
    bladderLabel[np.nonzero(bladder)] = 1
    rectumLabel = np.zeros((PTV.shape))
    rectumLabel[np.nonzero(rectum)] = 1
    PTVLabel = np.zeros((PTV.shape))
    PTVLabel[np.nonzero(PTV)] = 1
    return targetLabelFinal, bladderLabel, rectumLabel, PTVLabel

targetLabelFinal, bladderLabel, rectumLabel, PTVLabel = loadMask(data_path2)

print(targetLabelFinal.shape, bladderLabel.shape, rectumLabel.shape)

doseMatrix = loadDoseMatrix(data_path)
print('doseMatrixShape',doseMatrix.shape)

def ProcessDmat(doseMatrix, targetLabels, bladderLabel, rectumLabel):
    x = np.ones((doseMatrix.shape[1],))
    MPTVtemp = doseMatrix[targetLabels == 1, :]
    DPTV = MPTVtemp.dot(x)
    MPTV = MPTVtemp[DPTV != 0,:]
    MBLAtemp = doseMatrix[targetLabels == 2, :]
    DBLA = MBLAtemp.dot(x)
    MBLA = MBLAtemp[DBLA != 0,:]
    MRECtemp = doseMatrix[targetLabels == 3, :]
    DREC = MRECtemp.dot(x)
    MREC = MRECtemp[DREC != 0,:]

    MBLAtemp1 = doseMatrix[bladderLabel == 1, :]
    DBLA1 = MBLAtemp1.dot(x)
    MBLA1 = MBLAtemp1[DBLA1 != 0,:]
    MRECtemp1 = doseMatrix[rectumLabel == 1, :]
    DREC1 = MRECtemp1.dot(x)
    MREC1 = MRECtemp1[DREC1 != 0,:]
    return MPTV, MBLA, MREC, MBLA1, MREC1

MPTV, _, _, MBLA1, MREC1 = ProcessDmat(doseMatrix, targetLabelFinal, bladderLabel, rectumLabel)

print("MBLA1", MBLA1.shape)
print("MREC1", MREC1.shape)
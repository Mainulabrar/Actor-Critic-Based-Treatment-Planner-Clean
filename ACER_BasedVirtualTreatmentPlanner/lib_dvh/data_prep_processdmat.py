# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 10:48:28 2019

@author: S191914
"""
import h5sparse
import h5py
import numpy as np
from scipy.sparse import vstack

data_path = '/data2/tensorflow_utsw/dose_deposition/prostate_dijs/f_dijs/001.hdf5'
data_path2 = '/data2/tensorflow_utsw/dose_deposition/plostate_dijs/f_masks/001.h5'

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

def loadMask(filename):
    mask = h5py.File(filename,'r')
    dosemask = mask['oar_ptvs']['dose']
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
    print("np.count_nonzero(targetLabelFinal == 2)",np.count_nonzero(targetLabelFinal == 2))
    targetLabelFinal[np.nonzero(rectum)] = 3
    print("np.count_nonzero(targetLabelFinal == 2)",np.count_nonzero(targetLabelFinal == 2))
    print("np.count_nonzero(targetLabelFinal == 3)",np.count_nonzero(targetLabelFinal == 3))
    targetLabelFinal[np.nonzero(PTV)] = 1
    print("np.count_nonzero(targetLabelFinal == 2)",np.count_nonzero(targetLabelFinal == 2))
    print("np.count_nonzero(targetLabelFinal == 3)",np.count_nonzero(targetLabelFinal == 3))
    bladderLabel = np.zeros((PTV.shape))
    bladderLabel[np.nonzero(bladder)] = 1
    rectumLabel = np.zeros((PTV.shape))
    rectumLabel[np.nonzero(rectum)] = 1
    PTVLabel = np.zeros((PTV.shape))
    PTVLabel[np.nonzero(PTV)] = 1
    return targetLabelFinal, bladderLabel, rectumLabel, PTVLabel


def ProcessDmat(doseMatrix, targetLabels, bladderLabel, rectumLabel):
    print("doseMatrix",doseMatrix)
    print("doseMatrix.shape",doseMatrix.shape)
    print("doseMatrix.shape[1]",doseMatrix.shape[1])
    x = np.ones((doseMatrix.shape[1],))
    print("targetLabels",targetLabels)
    print("targetLabels.shape",targetLabels.shape)
    MPTVtemp = doseMatrix[targetLabels == 1, :]
    print("MPTVtemp",MPTVtemp)
    print("MPTVtemp[0]",MPTVtemp[0])
    print("MPTVtemp.getrow(0)",MPTVtemp.getrow(0))
    print("MPTVtemp.shape",MPTVtemp.shape)
    DPTV = MPTVtemp.dot(x)
    print("DPTVtemp",DPTV)
    print("DPTVtemp.shape",DPTV.shape)
    MPTV = MPTVtemp[DPTV != 0,:]
    print("MPTVtemp[DPTV != 0,:].shape", MPTV.shape)
    MBLAtemp = doseMatrix[targetLabels == 2, :]
    DBLA = MBLAtemp.dot(x)
    MBLA = MBLAtemp[DBLA != 0,:]
    print("MBLAtemp[DBLA != 0,:].shape", MBLA.shape)
    MRECtemp = doseMatrix[targetLabels == 3, :]
    DREC = MRECtemp.dot(x)
    MREC = MRECtemp[DREC != 0,:]
    print("MRECtemp[DREC != 0,:].shape", MREC.shape)

    MBLAtemp1 = doseMatrix[bladderLabel == 1, :]
    DBLA1 = MBLAtemp1.dot(x)
    MBLA1 = MBLAtemp1[DBLA1 != 0,:]
    print("MBLAtemp1[DBLA1 != 0,:].shape", MBLA1.shape)
    MRECtemp1 = doseMatrix[rectumLabel == 1, :]
    DREC1 = MRECtemp1.dot(x)
    MREC1 = MRECtemp1[DREC1 != 0,:]
    print("MRECtemp1[DREC1 != 0,:].shape", MREC1.shape)
    return MPTV, MBLA, MREC, MBLA1, MREC1
    
    
Dmat = loadDoseMatrix(data_path)
targetLabelFinal, bladderLabel, rectumLabel, PTVLabel = loadMask(data_path2)
ProcessDmat(Dmat, targetLabelFinal, bladderLabel, rectumLabel)
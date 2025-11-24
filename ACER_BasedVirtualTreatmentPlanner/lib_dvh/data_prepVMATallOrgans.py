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

    numbers = [f"{i:03d}" for i in range(0, 359, 2)]
    # numbers = ['000', '032', '064', '096', '296', '264', '328']
    print(numbers)
    Dmat = test['Dij']['000'].value

    for i in range(1,len(numbers)):
        print('i', i)
        temp = test['Dij'][numbers[i]].value
        Dmat = vstack([Dmat, temp])

    Dmat = Dmat.transpose()
    print('DmatShape',Dmat.shape)
    return Dmat

def loadMask(filename, OARNames):
    mask = h5py.File(filename, 'r')
    dosemask = mask['oar_ptvs']['dose']
    dosemask = np.reshape(dosemask, (dosemask.shape[0] * dosemask.shape[1] * dosemask.shape[2],), order='C')
    PTVtemp = mask['oar_ptvs']['ptv']
    print('PTVTempShape',PTVtemp.shape)
    PTVtemp = np.reshape(PTVtemp, (PTVtemp.shape[0] * PTVtemp.shape[1] * PTVtemp.shape[2],), order='C')
    print('PTVTempShapeAfterflattening', PTVtemp.shape)
    print('dosemaskshape', dosemask.shape)
    PTV = PTVtemp[np.nonzero(dosemask)]
    print('PTVShape', PTV.shape)
    OARs = []

    for file in OARNames:
        OARmasktemp = mask['oar_ptvs'][file]
        OARmasktemp = np.reshape(OARmasktemp, (OARmasktemp.shape[0] * OARmasktemp.shape[1] * OARmasktemp.shape[2],), order='C')
        OARs.append(OARmasktemp[np.nonzero(dosemask)])

    OARLabels = []
    targetLabelFinal = np.zeros((PTV.shape)) 

    for i in range(len(OARs)):
        targetLabelFinal[np.nonzero(OARs[i])] = i+2
        OARLabel =  np.zeros((OARs[i].shape))
        OARLabel[np.nonzero(OARs[i])] = 1
        OARLabels.append(OARLabel)

    targetLabelFinal[np.nonzero(PTV)] = 1    

    PTVLabel = np.zeros((PTV.shape))
    PTVLabel[np.nonzero(PTV)] = 1  

    return targetLabelFinal, PTVLabel, OARLabels

# targetLabelFinal, PTVLabel, OARLabels = loadMask(data_path2, ['bladder', 'rectum'])
# print(targetLabelFinal.shape, PTVLabel.shape, OARLabels[0].shape, OARLabels[1].shape)

# doseMatrix = loadDoseMatrix(data_path)
# print('doseMatrixShape',doseMatrix.shape)
# print("OARLabels Shape", OARLabels[0].shape, OARLabels[1].shape)

def ProcessDmat(doseMatrix, targetLabels, OARLabels):
    x = np.ones((doseMatrix.shape[1],))
    MPTVtemp = doseMatrix[targetLabels == 1, :]
    print('MPTVtempShape',MPTVtemp.shape)
    DPTV = MPTVtemp.dot(x)
    print('DPTVtempShape',DPTV.shape)
    MPTV = MPTVtemp[DPTV != 0,:]
    print('MPTVshape',MPTV.shape)

    MOARs = []

    MOAR1s = []

    for i in range(len(OARLabels)):
        MOARtemp = doseMatrix[targetLabels == i+2]
        DOAR = MOARtemp.dot(x)
        MOAR = MOARtemp[DOAR != 0]
        MOARs.append(MOAR)

        MOARtemp1 = doseMatrix[OARLabels[i] == 1, :]
        DOAR1 = MOARtemp1.dot(x)
        MOAR1 = MOARtemp1[DOAR1 != 0,:]
        MOAR1s.append(MOAR1)


    return MPTV, MOARs, MOAR1s

# MPTV, MOARs, MOAR1s = ProcessDmat(doseMatrix, targetLabelFinal, OARLabels)
# print("MOARs1 shape", MOAR1s[0].shape, MOAR1s[1].shape)

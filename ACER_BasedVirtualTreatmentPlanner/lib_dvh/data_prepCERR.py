# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 10:48:28 2019

@author: S191914
"""
import h5sparse
import h5py
import numpy as np
from scipy.sparse import vstack
import scipy.sparse as sp
from scipy.io import loadmat


def getOnlyNoneroDij(A):
    nonzeroRowMask = np.array(A.sum(axis =1)).ravel() != 0

    return A[nonzeroRowMask,:]


def loadDoseMatrix(filename):
    File = h5sparse.File(filename,'r')
    print("Keys in the file:", File['ABody'].keys())
    data = File['ABody']['data'][()]
    ir = File['ABody']['ir'][()]
    jc = File['ABody']['jc'][()]
    print(File['shape'][()])
    # nCols = len(jc)-1
    # nRows = int(ir.max()+1)
    nRows = int(File['shape'][()][0][0])
    nCols = int(File['shape'][()][1][0])
    print(nRows)

    Dmat = sp.csc_matrix((data, ir, jc), shape = (nRows, nCols))
    print(Dmat.shape)
    DmatNonzero = getOnlyNoneroDij(Dmat)
    print('nonzeroshape',DmatNonzero.shape)
    return Dmat, Dmat.shape, DmatNonzero

# _,_,doseMatrix = loadDoseMatrix("/home/mainul/CERRdata/1DijBody.mat")
# print('doseMatrixshape', doseMatrix.shape)

def loadMask(filenameDij, filenamePTV, filenameOAR):
    mask = loadmat(filenamePTV)['finalMask'][()]

    print('PTVMaskShape',mask.shape)
    Dmat, DmatShape, _ = loadDoseMatrix(filenameDij)
    Dose = Dmat.dot(np.ones((DmatShape[1])))
    print(Dose.shape)
    Dose3D = Dose.reshape((mask.shape), order = 'F')
    Dose3D = (Dose3D != 0).astype(np.uint8)
    print(Dose3D.shape)
    dosemask = Dose3D
    dosemask = np.reshape(dosemask, (dosemask.shape[0] * dosemask.shape[1] * dosemask.shape[2],), order='F')
    PTVtemp = mask
    print('PTVTempShape',PTVtemp.shape)
    PTVtemp = np.reshape(PTVtemp, (PTVtemp.shape[0] * PTVtemp.shape[1] * PTVtemp.shape[2],), order='F')
    print('PTVTempShapeAfterflattening', PTVtemp.shape)
    print('dosemaskshape', dosemask.shape)
    PTV = PTVtemp[np.nonzero(dosemask)]
    print('PTVShape', PTV.shape)
    OARs = []

    for file in filenameOAR:
        mask = loadmat(file)['finalMask'][()]
        masktemp = np.reshape(mask, (mask.shape[0] * mask.shape[1] * mask.shape[2],), order='F')
        OARs.append(masktemp[np.nonzero(dosemask)])

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

# targetLabels, PTVLabel, OARLabels = loadMask("/home/mainul/CERRdata/1DijBody.mat" , "/home/mainul/CERRdata/1structure_PTV.mat", ["/home/mainul/CERRdata/1structure_Bladder.mat", "/home/mainul/CERRdata/1structure_Rectum.mat"])

# print('targetLabels', targetLabels.shape)

# print(targetLabelFinal.shape)
# for i in range(len(OARLabels)):
#     print(OARLabels[i].shape)

# def loadMask(filename1, filename2, filename3, filename4, filename5, filename6,filename7, filename8, filename9):
#     mask = h5py.File(filename,'r')
#     dosemask = mask['oar_ptvs']['dose']
#     dosemask = np.reshape(dosemask, (dosemask.shape[0] * dosemask.shape[1] * dosemask.shape[2],), order='C')
#     PTVtemp = mask['oar_ptvs']['ptv']
#     PTVtemp = np.reshape(PTVtemp, (PTVtemp.shape[0] * PTVtemp.shape[1] * PTVtemp.shape[2],), order='C')
#     PTV = PTVtemp[np.nonzero(dosemask)]
#     bladdertemp = mask['oar_ptvs']['bladder']
#     bladdertemp = np.reshape(bladdertemp, (bladdertemp.shape[0] * bladdertemp.shape[1] * bladdertemp.shape[2],), order='C')
#     bladder = bladdertemp[np.nonzero(dosemask)]
#     rectumtemp = mask['oar_ptvs']['rectum']
#     rectumtemp = np.reshape(rectumtemp, (rectumtemp.shape[0] * rectumtemp.shape[1] * rectumtemp.shape[2],), order='C')
#     rectum = rectumtemp[np.nonzero(dosemask)]
#     targetLabelFinal = np.zeros((PTV.shape))
#     targetLabelFinal[np.nonzero(bladder)] = 2
#     targetLabelFinal[np.nonzero(rectum)] = 3
#     targetLabelFinal[np.nonzero(PTV)] = 1
#     bladderLabel = np.zeros((PTV.shape))
#     bladderLabel[np.nonzero(bladder)] = 1
#     rectumLabel = np.zeros((PTV.shape))
#     rectumLabel[np.nonzero(rectum)] = 1
#     PTVLabel = np.zeros((PTV.shape))
#     PTVLabel[np.nonzero(PTV)] = 1
#     return targetLabelFinal, bladderLabel, rectumLabel, PTVLabel


# def ProcessDmat(doseMatrix, targetLabels, bladderLabel, rectumLabel):
#     x = np.ones((doseMatrix.shape[1],))
#     MPTVtemp = doseMatrix[targetLabels == 1, :]
#     DPTV = MPTVtemp.dot(x)
#     MPTV = MPTVtemp[DPTV != 0,:]
#     MBLAtemp = doseMatrix[targetLabels == 2, :]
#     DBLA = MBLAtemp.dot(x)
#     MBLA = MBLAtemp[DBLA != 0,:]
#     MRECtemp = doseMatrix[targetLabels == 3, :]
#     DREC = MRECtemp.dot(x)
#     MREC = MRECtemp[DREC != 0,:]

#     MBLAtemp1 = doseMatrix[bladderLabel == 1, :]
#     DBLA1 = MBLAtemp1.dot(x)
#     MBLA1 = MBLAtemp1[DBLA1 != 0,:]
#     MRECtemp1 = doseMatrix[rectumLabel == 1, :]
#     DREC1 = MRECtemp1.dot(x)
#     MREC1 = MRECtemp1[DREC1 != 0,:]
#     return MPTV, MBLA, MREC, MBLA1, MREC1

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

# MPTV, MBLA, MREC, MBLA1, MREC1 = ProcessDmat(doseMatrix, targetLabels, OARLabels)
# print("MPTVshpae", MPTV.shape)
    # MBLAtemp = doseMatrix[targetLabels == 2, :]
    # DBLA = MBLAtemp.dot(x)
    # MBLA = MBLAtemp[DBLA != 0,:]
    # MRECtemp = doseMatrix[targetLabels == 3, :]
    # DREC = MRECtemp.dot(x)
    # MREC = MRECtemp[DREC != 0,:]

    # MBLAtemp1 = doseMatrix[bladderLabel == 1, :]
    # DBLA1 = MBLAtemp1.dot(x)
    # MBLA1 = MBLAtemp1[DBLA1 != 0,:]
    # MRECtemp1 = doseMatrix[rectumLabel == 1, :]
    # DREC1 = MRECtemp1.dot(x)
    # MREC1 = MRECtemp1[DREC1 != 0,:]
    # return MPTV, MBLA, MREC, MBLA1, MREC1
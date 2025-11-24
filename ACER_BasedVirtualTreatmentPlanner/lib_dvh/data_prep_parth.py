# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 10:48:28 2019

@author: S191914
"""
import h5sparse
import h5py
import numpy as np
from scipy.sparse import vstack
import scipy.io

data_path = '/data2/tensorflow_utsw/dose_deposition/prostate_dijs/f_dijs/001.hdf5'
# data_path2 = '/home/parvat/acer VTPN/001.h5'
data_path2 = "/home/mainul1/ACER_basic/002_structure_mask.h5"
data_path3 = "/home/mainul1/ACER_basic/002_data_mask.h5"

def loadDoseMatrix(filename):
    file = h5py.File(filename, 'r')
    print("file", file)
    print(list(file.keys()))
    
    # Access a specific group or dataset
    group = file['Dij']
    group2 = file['Sij']
    
    print("group", group)
    print("list(group.keys()): ",list(group.keys()))
    print("group2", group2)
    print("list(group2.keys()): ",list(group2.keys()))
    
    group3 = group['000']
    print("group3", group3)
    print("list(group.keys()): ",list(group3.keys()))
    
    group4 = group3['data']
    print("group4", group4)

    
    group5 = group3['indices']
    print("group5", group5)

    
    group6 = group3['indptr']
    print("group6", group6)


    # with h5py.File(filename,'r') as f:
        # f.visit(print)
    
    # test = h5sparse.File(filename,'r')
    # print(test)
    
    # Dmat = test['Dij']['000'].value
    # print("Dmat",Dmat)
    
    
    # temp = test['Dij']['032'].value
    # print("temp",temp)
    # Dmat = vstack([Dmat, temp])
    # print("Dmat_edited",Dmat)
    
    
    # temp = test['Dij']['064'].value
    # Dmat = vstack([Dmat, temp])
    # temp = test['Dij']['096'].value
    # Dmat = vstack([Dmat, temp])
    # temp = test['Dij']['296'].value
    # Dmat = vstack([Dmat, temp])
    # temp = test['Dij']['264'].value
    # Dmat = vstack([Dmat, temp])
    # temp = test['Dij']['328'].value
    # Dmat = vstack([Dmat, temp])
    
    
    # Dmat = Dmat.transpose()
    # print("Transposed_Dmat",Dmat)
    # return Dmat

def loadMask(filename1, filename2):
    print(filename1)
    print(filename2)
    file1 = h5py.File(filename1, 'r')
    file2 = h5py.File(filename2, 'r')
    print("Groups and datasets in the .h5 file:")
    print(list(file1.keys()))
    print(list(file2.keys()))

    # Access a specific group or dataset    
    # For Parth's 002_structure_mask.h5
    group = file1['oar_ptv']
    # For Parth's 002_data_mask.h5
    group2 = file2['dose_mask']
    

    # Print information about the group or dataset
    print("Information about the group:")
    print("group: ",group)
    print("list(group.keys()): ",list(group.keys()))
    print("group2: ",group2)
    # print("list(group2.keys()): ",list(group2.keys()))

    
    #For Parth's  002_structure_mask.h5
    print('group["PTV_56"]: ', group["PTV_56"])
    print('group["PTV_68"]: ', group["PTV_68"])
    print('group["Bladder"]: ', group["Bladder"])
    print('group["Rectum"]: ', group["Rectum"])
    print("========================================")
    
    

    dosemask = group2
    print("dosemask_before",dosemask)
    dosemask = np.reshape(dosemask, (dosemask.shape[0] * dosemask.shape[1] * dosemask.shape[2],), order='C')
    print("dosemask = np.reshape(dosemask, (dosemask.shape[0] * dosemask.shape[1] * dosemask.shape[2],), order='C'",dosemask)
    print("dosemask.shape",dosemask.shape )
    print("========================================")
    
    

    PTVtemp = group["PTV_56"]
    print("PTVtemp_before",PTVtemp )
    PTVtemp = np.reshape(PTVtemp, (PTVtemp.shape[0] * PTVtemp.shape[1] * PTVtemp.shape[2],), order='C')
    print("PTVtemp = np.reshape(PTVtemp, (PTVtemp.shape[0] * PTVtemp.shape[1] * PTVtemp.shape[2],), order='C')", PTVtemp)
    print("PTVtemp.shape", PTVtemp.shape)
    PTV = PTVtemp[np.nonzero(dosemask)]
    print("PTV = PTVtemp[np.nonzero(dosemask)]", PTV)
    print("PTV.shape", PTV.shape)
    print("=========================")
    
    
    bladdertemp = group["Bladder"]
    print("bladdertemp_before",bladdertemp )
    bladdertemp = np.reshape(bladdertemp, (bladdertemp.shape[0] * bladdertemp.shape[1] * bladdertemp.shape[2],), order='C')
    print("bladdertemp = np.reshape(bladdertemp, (bladdertemp.shape[0] * bladdertemp.shape[1] * bladdertemp.shape[2],), order='C')",bladdertemp)
    bladder = bladdertemp[np.nonzero(dosemask)]
    print("bladder = bladdertemp[np.nonzero(dosemask)]",bladder)
    print("bladder.shape",bladder.shape)
    print("=========================")
    
    rectumtemp = group["Rectum"]
    print("rectumtemp_before",rectumtemp )
    rectumtemp = np.reshape(rectumtemp, (rectumtemp.shape[0] * rectumtemp.shape[1] * rectumtemp.shape[2],), order='C')
    print("rectumtemp = np.reshape(rectumtemp, (rectumtemp.shape[0] * rectumtemp.shape[1] * rectumtemp.shape[2],), order='C')",rectumtemp)
    rectum = rectumtemp[np.nonzero(dosemask)]
    print("rectum = rectumtemp[np.nonzero(dosemask)]",rectum)
    print("rectum.shape",rectum.shape)
    print("=========================")
    
    
    targetLabelFinal = np.zeros((PTV.shape))
    print("targetLabelFinal = np.zeros((PTV.shape))",targetLabelFinal)
    print("targetLabelFinal.shape",targetLabelFinal.shape)
    targetLabelFinal[np.nonzero(bladder)] = 2
    targetLabelFinal[np.nonzero(rectum)] = 3
    targetLabelFinal[np.nonzero(PTV)] = 1
    print("=========================")
    
    
    bladderLabel = np.zeros((PTV.shape))
    print("bladderLabel = np.zeros((PTV.shape))",bladderLabel)
    print("bladderLabel.shape",bladderLabel.shape)
    bladderLabel[np.nonzero(bladder)] = 1
    rectumLabel = np.zeros((PTV.shape))
    rectumLabel[np.nonzero(rectum)] = 1
    PTVLabel = np.zeros((PTV.shape))
    PTVLabel[np.nonzero(PTV)] = 1
    print("=========================")
    return targetLabelFinal, bladderLabel, rectumLabel, PTVLabel
    
    




    # mask = h5py.File(filename,'r')
    # dosemask = mask['oar_ptvs']['dose']
    # dosemask = np.reshape(dosemask, (dosemask.shape[0] * dosemask.shape[1] * dosemask.shape[2],), order='C')
    # PTVtemp = mask['oar_ptvs']['ptv']
    # PTVtemp = np.reshape(PTVtemp, (PTVtemp.shape[0] * PTVtemp.shape[1] * PTVtemp.shape[2],), order='C')
    # print("PTVtemp", PTVtemp.shape)
    # PTV = PTVtemp[np.nonzero(dosemask)]
    # PTV2 = PTV[np.nonzero(PTV)]
    # print("PTV2.shape", PTV2.shape) # overlap between dosemask and PTV
    # print("PTV.shape", PTV.shape) # length of dosemask, only belonging to PTV is 1, and those that do not belong is 0.
    # bladdertemp = mask['oar_ptvs']['bladder']
    # bladdertemp = np.reshape(bladdertemp, (bladdertemp.shape[0] * bladdertemp.shape[1] * bladdertemp.shape[2],), order='C')
    # bladder = bladdertemp[np.nonzero(dosemask)]
    # print("PTV.shape", bladder.shape)
    # rectumtemp = mask['oar_ptvs']['rectum']
    # rectumtemp = np.reshape(rectumtemp, (rectumtemp.shape[0] * rectumtemp.shape[1] * rectumtemp.shape[2],), order='C')
    # rectum = rectumtemp[np.nonzero(dosemask)]
    # print("PTV.shape", rectum.shape)
    # targetLabelFinal = np.zeros((PTV.shape))
    # targetLabelFinal[np.nonzero(bladder)] = 2
    # targetLabelFinal[np.nonzero(rectum)] = 3
    # targetLabelFinal[np.nonzero(PTV)] = 1
    # bladderLabel = np.zeros((PTV.shape))
    # bladderLabel[np.nonzero(bladder)] = 1
    # rectumLabel = np.zeros((PTV.shape))
    # rectumLabel[np.nonzero(rectum)] = 1
    # PTVLabel = np.zeros((PTV.shape))
    # PTVLabel[np.nonzero(PTV)] = 1
    # return targetLabelFinal, bladderLabel, rectumLabel, PTVLabel


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

# loadDoseMatrix(data_path)
loadMask(data_path2, data_path3)
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

def loadDoseMatrix(filename):
    test = h5sparse.File(filename,'r')
    Dmat = test['Dij']['000'].value
    print("Dmat.shape",Dmat.shape)
    print("Dmat.indices.max()",Dmat.indices.max())
    temp = test['Dij']['032'].value
    print("temp.shape",temp.shape)
    print("temp.indices.max()",temp.indices.max())
    Dmat = vstack([Dmat, temp])
    print("Dmat_edited",Dmat)
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
    Dmat = Dmat.transpose()
    print("Transposed_Dmat",Dmat)
    # return Dmat

def loadMask(filename):
    file = h5py.File(filename, 'r')
    print("Groups and datasets in the .h5 file:")
    print(list(file.keys()))

    # Access a specific group or dataset
    group = file['oar_ptvs']
    group2 = file['fluence_grp']

    # Print information about the group or dataset
    print("Information about the group:")
    print("group: ",group)
    print("list(group.keys()): ",list(group.keys()))
    print("group2: ",group2)
    print("list(group2.keys()): ",list(group2.keys()))

    # print('group["dose"]: ',group["dose"])
    # print('group["ptv"]: ', group["ptv"])
    # print('group["bladder"]: ', group["bladder"])
    # print('group["rectum"]: ', group["rectum"])

    # dosemask = group["dose"]
    # print("dosemask_before",dosemask)
    # dosemask = np.reshape(dosemask, (dosemask.shape[0] * dosemask.shape[1] * dosemask.shape[2],), order='C')
    # print("dosemask_after",dosemask)

    PTVtemp = group["ptv"]
    print("PTVtemp_before",PTVtemp )
    PTVtemp = np.reshape(PTVtemp, (PTVtemp.shape[0] * PTVtemp.shape[1] * PTVtemp.shape[2],), order='C')
    print("PTVtemp_after", PTVtemp)
    print("PTVtemp_after", PTVtemp.shape)


    dosemask = group["dose"]
    data = {"dosemask": dosemask}
    values = dosemask_3d[:]
    # matrix = np.reshape(values, (512,512,429))
    scipy.io.savemat("dosemask_3d_h5.mat", data)
    print("dosemask", dosemask)
    dosemask = np.reshape(dosemask, (dosemask.shape[0] * dosemask.shape[1] * dosemask.shape[2],), order='C')
    print("np.nonzero(dosemask)", np.nonzero(dosemask))
    print("dosemask.shape", dosemask.shape)
    print("dosemask", dosemask)

    PTV2 = PTVtemp[np.nonzero(PTVtemp)]
    print("PTV2.shape", PTV2.shape)
    PTV = PTVtemp[np.nonzero(dosemask)]
    print("PTV.shape", PTV.shape)
    print("=========================")




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

loadDoseMatrix(data_path)
# loadDoseMatrix("C:/Users/psapkota/PycharmProjects/Actor Critic based VTPN/data/data/GPU2AC/data/data/dose_deposition3/f_dijs/001.hdf5")
# loadMask("C:/Users/psapkota/PycharmProjects/Actor Critic based VTPN/data/data/GPU2AC/data/data/dose_deposition3/f_masks/001.h5")
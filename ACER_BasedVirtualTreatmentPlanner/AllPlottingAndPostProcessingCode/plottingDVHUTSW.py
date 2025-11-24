import numpy as np
import matplotlib.pyplot as plt
import re
import fnmatch
import os
import h5sparse
import h5py
from scipy.sparse import vstack
from scipy.sparse import csc_matrix
import matplotlib.colors as mcolors
import sys
sys.path.append('/home/mainul/Actor-critic-based-treatment-planning/acer_VTPN-12-18-23/')
import numpy.linalg as LA
import time
import os

INPUT_SIZE = 100

def MinimizeDoseOAR_dvh(MPTV, MBLA, MREC,tPTV, tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, xVec, gamma,pdose,maxiter):
    # treatment planning optimization in DVH-based scheme
    beta=2
    lambdaBLA = lambdaBLA/lambdaPTV
    lambdaREC = lambdaREC/lambdaPTV
    # xVec = np.ones((MPTV.shape[1],))
    DPTV = MPTV.dot(xVec)
    # print('2nd DPTV Index', DPTV[1])
    DPTV1 = np.sort(DPTV)
    # print('2nd Sorted DPTV Index', DPTV1[1])
    posi = int(round(0.05 * DPTV1.shape[0]))
    D95 = DPTV1[posi]
#    MPTV95 = MPTV[DPTV>=D95,:]
    factor = pdose / D95
    xVec = xVec * factor
    y = MPTV.dot(xVec)
    MPTVT = MPTV.transpose()
    DPTV = MPTV.dot(xVec)
    for iter in range(maxiter):
        xVec_old = xVec
        DPTV1 = np.sort(DPTV)
        posi = int(round((1 - VPTV) * DPTV1.shape[0]))-1
        if posi < 0:###### what was missing
            posi = 0###### what was missing
        DPTVV = DPTV1[posi]

        # posi = int(round(0.05 * DPTV1.shape[0]))-1
        # D95 = DPTV1[posi]
        # MPTV95 = MPTV[DPTV >= D95, :]
        # DPTV95 = MPTV95.dot(xVec)
        DBLA = MBLA.dot(xVec)
        DBLA1 = np.sort(DBLA)
        # print('1st DOAR Bla Index', DBLA[1])
        # print('1st sorted DOAR Bla Index', DBLA1[1])
        posi = int(round((1 - VBLA) * DBLA1.shape[0]))-1
        if posi < 0:
            posi = 0
        DBLAV = DBLA1[posi]
        DREC = MREC.dot(xVec)
        DREC1 = np.sort(DREC)
        # print('1st DOAR Rec Index', DREC[1])
        # print('1st sorted DOAR Rec Index', DREC1[1])
        posi = int(round((1 - VREC) * DREC1.shape[0]))-1
        if posi < 0:
            posi = 0
        DRECV = DREC1[posi]

        MPTVV =  MPTV[DPTV>=DPTVV,:]
        temp= DPTV[DPTV>=DPTVV]
        if np.max(temp) > pdose* tPTV:
            MPTV1 = MPTVV[temp > pdose*tPTV, :]
            targetPTV1 = pdose*tPTV*np.ones((MPTV1.shape[0],))
            MPTV1T = MPTV1.transpose()
            temp1 = MPTV1.dot(xVec)
            temp1 = MPTV1T.dot(temp1)
            temp1 = temp1 * 1/MPTV1.shape[0]
            b1 = MPTV1T.dot(targetPTV1) / MPTV1.shape[0]
        else:
            temp1 = np.zeros((xVec.shape))
            b1 = np.zeros((xVec.shape))
            tempp1 = np.zeros((xVec.shape))
        tempptv = temp

        temp2 = MPTV.dot(xVec)
        temp2 = beta*MPTVT.dot(temp2)/y.shape[0]
        b2 =  beta*MPTVT.dot(y)/y.shape[0]


        MBLAV = MBLA[DBLA>=DBLAV,:]
        temp = DBLA[DBLA>=DBLAV]
        # print('first temp bla index', temp[1])
        # print('max temp bla index', np.max(temp))
        if np.max(temp) > pdose * tBLA:
            MBLA1 = MBLAV[temp > pdose * tBLA, :]
            targetBLA1 = pdose*tBLA*np.ones((MBLA1.shape[0],))
            # print('first targetOARs bla', targetBLA1[1])
            MBLA1T = MBLA1.transpose()
            temp3 = MBLA1.dot(xVec)
            # print('temp after Mult xVec 1st bla', temp3[1])
            temp3 = MBLA1T.dot(temp3)
            # print('temp after Mult xVec 2nd bla', temp3[1])
            temp3 = temp3 * lambdaBLA/MBLA1.shape[0]
            # print('temp after Mult xVec 3rd bla', temp3[1])
            b3 = lambdaBLA * MBLA1T.dot(targetBLA1) / max(MBLA1.shape[0], 1)
        else:
            temp3 = np.zeros((xVec.shape))
            b3 = np.zeros((xVec.shape))
            tempp3 = np.zeros((xVec.shape))
        tempbla = temp
        # print('temp3 shape' , temp3.shape)
        # print('b sum', np.sum(b3))
        # print('temp after Mult xVec final bla', np.sum(temp3))

        MRECV = MREC[DREC >= DRECV, :]
        temp = DREC[DREC >= DRECV]
        # print('first temp rec index', temp[1])
        # print('max temp bla index', np.max(temp))
        if np.max(temp) > pdose * tREC:
            MREC1 = MRECV[temp > pdose * tREC, :]
            targetREC1 = pdose*tREC*np.ones((MREC1.shape[0],))
            # print('first targetOARs rec', targetREC1[1])
            MREC1T = MREC1.transpose()
            temp4 = MREC1.dot(xVec)
            # print('temp after Mult xVec 1st rec', temp4[1])
            temp4 = MREC1T.dot(temp4)
            # print('temp after Mult xVec 2nd rec', temp4[1])
            temp4 = temp4 * lambdaREC/MREC1.shape[0]
            # print('temp after Mult xVec 3rd rec', temp4[1])
            b4 = lambdaREC * MREC1T.dot(targetREC1) / MREC1.shape[0]
        else:
            temp4 = np.zeros((xVec.shape))
            b4 = np.zeros((xVec.shape))
            tempp4 = np.zeros((xVec.shape))
        temprec = temp
        # print('b sum', np.sum(b4))
        # print('temp after Mult xVec final rec', np.sum(temp4))

        templhs = temp1+temp2+temp3+temp4
        # print('templhs', np.sum(templhs))
        b = b1+b2+b3+b4-MPTVT.dot(gamma)
        # print('total b', np.sum(b))
        r = b - templhs
        # print('total r', np.sum(r))
        # if iter == 0:
        #     np.save(/)
            # np.save('/data4/tempsOriSum', temp3+temp4)
            # print('type',type(temp3))
            # np.save('/data4/tempsOri2', temp2)
            # np.save('/data4/tempsOri1', temp1)
            # np.save('/data4/bori.npz', b)
            # np.save('/data4/templhsori', temp1+temp2+temp3+temp4)
        p = r
        rsold = np.inner(r,r)

        # print('p shape', np.sum(p))
        # print()
        # print('iter', iter,  rsold)
        # print("rsold=", rsold, "iter=", iter, "=========================")  # this not

        if rsold>1e-10:
            for i in range(3):
                if np.max(tempptv) > pdose*tPTV :
                    tempp1 = MPTV1.dot(p)
                    tempp1 = MPTV1T.dot(tempp1)
                    tempp1 = tempp1 * 1 / MPTV1.shape[0]


                tempp2 = MPTV.dot(p)
                tempp2 = beta * MPTVT.dot(tempp2)/y.shape[0]

                if np.max(tempbla) > pdose * tBLA:
                    tempp3 = MBLA1.dot(p)
                    tempp3 = MBLA1T.dot(tempp3)
                    tempp3 = tempp3 * lambdaBLA / MBLA1.shape[0]
                    # print('tempp bla', np.sum(tempp3))

                if np.max(temprec) > pdose * tREC:
                    tempp4 = MREC1.dot(p)
                    tempp4 = MREC1T.dot(tempp4)
                    tempp4 = tempp4 * lambdaREC / MREC1.shape[0]
                    # print('tempp rec', np.sum(tempp4))

                Ap = tempp1 + tempp2 + tempp3 + tempp4
                # print('Ap',i , np.sum(Ap))
                pAp = np.inner(p, Ap)
                alpha = rsold / pAp
                xVec = xVec + alpha * p
                # print('xVecFinal', np.sum(xVec))
                xVec[xVec<0]=0
                r = r - alpha * Ap
                rsnew = np.inner(r, r)
                if np.sqrt(rsnew) < 1e-5:
                    break
                p = r + (rsnew / rsold) * p
                rsold = rsnew
        DPTV = MPTV.dot(xVec)
        y = (DPTV * beta/y.shape[0] + gamma) / (beta/y.shape[0])
        Dy = np.sort(y)
        posi = int(round(0.05 * Dy.shape[0]))
        D95 = Dy[posi]
        temp = np.zeros(y.shape)
        temp[y>=D95] = y[y>=D95]
        temp[temp<pdose] = pdose
        y[y>=D95] = temp[y>=D95]
        gamma = gamma + beta * (MPTV.dot(xVec)-y)/y.shape[0]

        if LA.norm(xVec - xVec_old, 2) / LA.norm(xVec_old, 2) < 5e-3:
            break
    DPTV = MPTV.dot(xVec)
    DPTV1 = np.sort(DPTV)
    posi = int(round(0.05 * DPTV1.shape[0]))
    D95 = DPTV1[posi]
    # print('Original D95', D95)
    factor = pdose / D95 # thresholidng
    xVec = xVec * factor
    converge = 1
    if iter == maxiter - 1:
        converge = 0
    # print("LOOKED HERE DAMON:",converge,iter)
    return xVec, iter

def runOpt_dvh(MPTV, MBLA, MREC,tPTV,tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, xVec,gamma,pdose,maxiter):
    # run optimization and generate DVH curves
    xVec, iter = MinimizeDoseOAR_dvh(MPTV, MBLA, MREC,tPTV,tBLA, tREC,lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, xVec,gamma,pdose,maxiter)
#    j = 0
    DPTV = MPTV.dot(xVec)
    DBLA = MBLA.dot(xVec)
    DREC = MREC.dot(xVec)
    DPTV = np.sort(DPTV)
    DPTV = np.flipud(DPTV)
    DBLA = np.sort(DBLA)
    DBLA = np.flipud(DBLA)
    DREC = np.sort(DREC)
    DREC = np.flipud(DREC)

    ## Plot DVH curve for optimized plan
    edge_ptv = np.zeros((INPUT_SIZE + 1,))
    edge_Full = np.zeros((INPUT_SIZE + 1,))
    edge_ptv[1:INPUT_SIZE + 1] = np.linspace(pdose, pdose * 1.15, INPUT_SIZE)
    edge_Full[1:INPUT_SIZE + 1] = np.linspace(0, pdose*1.2, INPUT_SIZE)
    (n_ptv, b) = np.histogram(DPTV, bins=edge_ptv)
    (n_ptvFull, b) = np.histogram(DPTV, bins=edge_Full)
    y_ptv = 1 - np.cumsum(n_ptv / len(DPTV), axis=0)
    y_ptvFull = 1 - np.cumsum(n_ptvFull / len(DPTV), axis=0)

    edge_bladder = np.zeros((INPUT_SIZE + 1,))
    edge_bladder[1:INPUT_SIZE + 1] = np.linspace(0.6 * pdose, 1.1 * pdose, INPUT_SIZE)
    (n_bladder, b) = np.histogram(DBLA, bins=edge_bladder)
    (n_bladderFull, b) = np.histogram(DBLA, bins=edge_Full)    
    y_bladder = 1 - np.cumsum(n_bladder / len(DBLA), axis=0)
    y_bladderFull = 1 - np.cumsum(n_bladderFull / len(DBLA), axis=0)

    edge_rectum = np.zeros((INPUT_SIZE + 1,))
    edge_rectum[1:INPUT_SIZE + 1] = np.linspace(0.6 * pdose, 1.1 * pdose, INPUT_SIZE)
    (n_rectum, b) = np.histogram(DREC, bins=edge_rectum)
    (n_rectumFull, b) = np.histogram(DREC, bins=edge_Full)    
    y_rectum = 1 - np.cumsum(n_rectum / len(DREC), axis=0)
    y_rectumFull = 1 - np.cumsum(n_rectumFull / len(DREC), axis=0)

    Y = np.zeros((INPUT_SIZE, 3))
    Y[:, 0] = y_ptv
    Y[:, 1] = y_bladder
    Y[:, 2] = y_rectum

    Yfull = np.zeros((INPUT_SIZE, 3))
    Yfull[:, 0] = y_ptvFull
    Yfull[:, 1] = y_bladderFull
    Yfull[:, 2] = y_rectumFull

    # plt.plot(edge_ptv[1:101], y_ptv, color = 'red' )
    # plt.plot(edge_bladder[1:101], Y[:,1], color = 'green')
    # plt.plot(edge_bladder[1:101], Y[:,2], color = 'blue')
    # # plt.plot(edge_bladder[1:101], Y[:,3], color = 'purple')
    # # plt.plot(edge_bladder[1:101], Y[:,4], color = 'yellow')
    # plt.show()
    # plt.close()

    Y = np.reshape(Y, (100 * 3,), order='F')

    return Y, iter, xVec, Yfull

patient_list = ['001', '008', '009', '010', '011', '013', '014', '015', '016', '017', '018', '020', '022', '023', '025', '026', '027', '028', '030', '031', '036', '037', '039', '042', '043', '045', '046', '054', '057', '061', '065', '066', '068', '070', '073', '074', '077', '080', '081', '083', '084', '085', '087', '088', '091', '093', '095', '097', '098']
#patient_list = [ '008',  '010',     '020',    '027',   '031',   '039', '042',  '046',  '061',  '070',    '084',  '087',  '092',  '095',  '098']


# data_result_path = '/data2/mainul/results_CORS1Paper/scratch6_30StepsNewParamenters3NewData1/dataWithPlanscoreRun/'
# data_result_path = '/data2/mainul/results_CORS1PaperDiffBeam/scratch6_30StepsNewParamenters3NewData1/dataWithPlanscoreRun/'
data_result_path = '/data2/mainul/results_CORS1PaperDiffBeam7/scratch6_30StepsNewParamenters3NewData1/dataWithPlanscoreRun/'
# data_result_path = '/data2/mainul/results_CORS1random6C/scratch6_30StepsNewParamenters3NewData1/dataWithPlanscoreRun/'
# graphSavePath = '/data2/mainul/DataAndGraph/'
# data_result_path = '/data4/mainul/MultiModalAI6Beam/Blafixed/PaperInitLSTM155500/dataWithPlanscoreRun/'
data_result_path = '/data4/mainul/MultiModalAI6Beam/Blafixed/PaperInitLSTM155500/dataWithPlanscoreRun/'
data_result_path = '/data4/mainul/MultiModalAI/Blafixed/PaperInitLSTM155500/dataWithPlanscoreRun/'
data_result_path = '/data4/mainul/MultiModalAI1/Blafixed/PaperInitLSTM155500/dataWithPlanscoreRun/'
data_result_path='/data4/mainul/MultiModalAIwithoutTPP/Blafixed/PaperInitLSTM155500/dataWithPlanscoreRun/'

def maximum_step(patientid):
    # data_path = '/data/mainul1/results_CORS1/scratch6_30StepsNewParamenters3NewData4/dataWithPlanscoreRun/'
    # data_path = '/data2/mainul/results_CORS1Paper/scratch6_30StepsNewParamenters3NewData1/dataWithPlanscoreRun/'
    # data_path = '/data2/mainul/results_CORS1PaperDiffBeam/scratch6_30StepsNewParamenters3NewData1/dataWithPlanscoreRun/'
    data_path = '/data2/mainul/results_CORS1PaperDiffBeam7/scratch6_30StepsNewParamenters3NewData1/dataWithPlanscoreRun/'    
    # data_path = '/data2/mainul/results_CORS1random6C/scratch6_30StepsNewParamenters3NewData1/dataWithPlanscoreRun/'
    pattern_base = "{}xDVHY*step*.npy"
    pattern = pattern_base.format(patientid)

    # Use fnmatch to filter filenames based on the pattern
    files = fnmatch.filter(os.listdir(data_path), pattern)

    # Print the matching filenames
    #print(files)
    captured_numbers = []

    # Iterate through the matching filenames
    for filename in files:
        # Use regular expression to extract the desired part
        match = re.search(r'^(\d+)xDVHY(\d+)step(\d+).npy$', filename)
        
        # Check if a match is found
        if match:
            captured_numbers.append(int(match.group(3)))
        else:
            print("No match found for:", filename)

    # Find the maximum value for this list
    max_value = max(captured_numbers)
    return max_value

def loadMask(filename):
    mask = h5py.File(filename,'r')
    dosemask = mask['oar_ptvs']['dose']
    dosemask = np.reshape(dosemask, (dosemask.shape[0] * dosemask.shape[1] * dosemask.shape[2],), order='C')
    PTVtemp = mask['oar_ptvs']['ptv']
    PTVtemp = np.reshape(PTVtemp, (PTVtemp.shape[0] * PTVtemp.shape[1] * PTVtemp.shape[2],), order='C')
    print('PTVtemp shape', PTVtemp.shape)
    PTV = PTVtemp[np.nonzero(dosemask)]
    print('PTV shape', PTV.shape)
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

import os 
epoch = 120499
evaluation_episodes = 1
max_episode_length = 31
# data_result_path='./data/data/Results/GPU2AC/general/cors_test_random_tpp1/3-2023-11-25/non-G/'
# data_result_path='/data/mainul1/results_CORS1/scratch6_30StepsNewParamenters3NewData4/dataWithPlanscoreRun/'
# data_result_path = '/data2/mainul/results_CORS1random6C/scratch6_30StepsNewParamenters3NewData1/dataWithPlanscoreRun/'
print(data_result_path)

## Next line is for CORT
# from data_prep_parth_complete_onceagain import loadDoseMatrix
# Next line is for TORTS
# from data_pret_TORTScor import loadDoseMatrix

from lib_dvh.data_prep import loadDoseMatrix, loadMask, ProcessDmat
# from lib_dvh.data_prep_diff_beam import loadDoseMatrix, loadMask, ProcessDmat
# id1 = 0

# The next line is enough for random
# id1 = patient_list[0]
# id1 = patient_list[22]
id1 = patient_list[1]
# print(patient_list[int((id1-1)//3)])
# Also add the following line for paper data
# id1 = patient_list[id1]

# doseMatrix = np.load(data_result_path+str(id1)+'doseMatrix' + str(epoch)  + 'step' + str(0)+'.npy',allow_pickle = True)
# doseMatrix = loadDoseMatrix('/home/mainul1/Actor-critic-based-treatment-planning/acer_VTPN-12-18-23/test_onceagain.hdf5')
# doseMatrix = loadDoseMatrix('/home/mainul1/Actor-critic-based-treatment-planning/acer_VTPN-12-18-23/test_TORTS.hdf5')
# doseMatrix = loadDoseMatrix('/media/mainul/Chi-Drives/128641-3-E21/var/lib/docker/overlay2/8fa0092f3bc6971c1752f734f8fd34c782377f383ffe6f5a55629a01d3b0f185/diff/home/exx/dose_deposition_full/prostate_dijs/f_dijs/008.hdf5')
# print(doseMatrix.shape)
#==================

# bladderLabel, rectumLabel, PTVLabel = loadMask()
# bladderLabel = np.load(data_result_path+str(id1)+'bladderLabel' + str(epoch)  + 'step' + str(0)+'.npy',allow_pickle = True)
# rectumLabel= np.load(data_result_path+str(id1)+'rectumLabel' + str(epoch)  + 'step' + str(0)+'.npy',allow_pickle = True)
# PTVLabel = np.load(data_result_path+str(id1)+'PTVLabel' + str(epoch)  + 'step' + str(0)+'.npy',allow_pickle = True)

data_path = '/media/mainul/Chi-Drives/128641-3-E21/var/lib/docker/overlay2/8fa0092f3bc6971c1752f734f8fd34c782377f383ffe6f5a55629a01d3b0f185/diff/home/exx/dose_deposition_full/prostate_dijs/f_dijs/'
data_path2 = '/media/mainul/Chi-Drives/128641-3-E21/var/lib/docker/overlay2/8fa0092f3bc6971c1752f734f8fd34c782377f383ffe6f5a55629a01d3b0f185/diff/home/exx/dose_deposition_full/plostate_dijs/f_masks/'

graphSavePath = '/data4/Proposal/'
os.makedirs(graphSavePath, exist_ok = True)


targetLabels, bladderLabel, rectumLabel, PTVLabel = loadMask(data_path2+str(id1)+'.h5')
doseMatrix = loadDoseMatrix(data_path+str(id1)+'.hdf5')

MPTV, _, _, MBLA, MREC = ProcessDmat(doseMatrix, targetLabels, bladderLabel, rectumLabel)

NpzFile = np.load(data_result_path+str(id1)+'tpptuning120499.npz')
NumberOfSteps = len(np.nonzero(NpzFile['l1'])[0])
print(NumberOfSteps)
# NpzFile = [1.0, 0.6, 0.6, 33.29565858, 0.6, 0.36, 0.1, 1.0, 1.0]


tPTV = NpzFile['l1'][20]
tBLA = NpzFile['l2'][20]
tREC = NpzFile['l3'][20]
lambdaPTV = NpzFile['l4'][20]
lambdaBLA = NpzFile['l5'][20]
lambdaREC = NpzFile['l6'][20]
VPTV = NpzFile['l7'][20]
VBLA = NpzFile['l8'][20]
VREC = NpzFile['l9'][20]
# Scores = NpzFile['l10']

# print('Scores', Scores)

# tPTV = NpzFile[0]
# tBLA = NpzFile[1]
# tREC = NpzFile[2]
# lambdaPTV = NpzFile[3]
# lambdaBLA = NpzFile[4]
# lambdaREC = NpzFile[5]
# VPTV = NpzFile[6]
# VBLA = NpzFile[7]
# VREC = NpzFile[8]

xVec = np.ones((MPTV.shape[1],))
gamma = np.zeros((MPTV.shape[0],))

pdose = 1
maxiter = 40
Y, iter, xVec, Yfull = runOpt_dvh(MPTV, MBLA, MREC,tPTV,tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, xVec,gamma,pdose,maxiter)


handles = []
labels = []
patient_num = ['008']

# DVH for each patient only few steps and making it full range
epoch = 120499
pdose = 1
Volume_axis = []
B_Volume_axis = []
R_Volume_axis = []
dose_axis_PTV = []
dose_axis_BLA = []
dose_axis_REC = []

colors = plt.cm.rainbow(np.linspace(0, 1, 3))
colors = ['red', 'green', 'blue']
linestyle = [':', '--', '-']

plt.rcParams["lines.linewidth"] = 5.5
plt.rcParams["axes.linewidth"] = 4.5
plt.rcParams["xtick.major.width"] = 4.5  # Major x-tick thickness
plt.rcParams["ytick.major.width"] = 4.5
plt.rcParams["xtick.major.size"] = 10  # Length of x-axis major ticks
plt.rcParams["ytick.major.size"] = 10

edge_Full = np.zeros((INPUT_SIZE + 1,))
edge_Full[1:INPUT_SIZE + 1] = np.linspace(0, pdose*1.2, INPUT_SIZE)

plt.figure(figsize=(12, 8))

for i in range(3):
    plt.plot(edge_Full[1:INPUT_SIZE + 1]*100, Yfull[:, i]*100, color = colors[i], linestyle = '-')


DoseBladder = [(65/79.5)*100, (70/79.5)*100, (75/79.5)*100, (80/79.5)*100]
DoseRectum = [(60/79.5)*100, (65/79.5)*100, (70/79.5)*100, (75/79.5)*100]
DosePTV = [100, 110]
Volume = [ 55, 40, 30, 20]
VolumePTV = [95, 1.0]

# DoseBlaToPlot = [(DoseBladder[i]/120)*100 for i in range(4)

plt.scatter(DoseBladder, Volume, marker = 's', color ='black', s = 200)
plt.scatter(DoseRectum, Volume, marker = '^', color ='black', s = 200)
plt.scatter(DosePTV, VolumePTV, marker = '*', color = 'black', s = 200)


plt.xticks(fontsize = 45)
plt.yticks(fontsize = 45)

# Add titles and labels
# plt.title('Adversarial Perturbation', fontsize = 14)
# plt.xlabel('Relative Volume (%)', fontsize = 40)
# plt.ylabel('Relative Volume (%)', fontsize = 40)

plt.xlabel('Relative Dose (%)', fontsize = 45)
plt.ylabel('Relative Volume (%)', fontsize = 45)
# plt.legend(ordered_handles, ordered_labels, loc='lower left', fontsize = 31)
plt.tight_layout()
plt.xlim(0, 120)
plt.ylim(0, 105)
# plt.xlabel('Dose Axis')
# plt.ylabel('Volume Axis')
# plt.title('DVH for Patient ID '+str(patientid))
# plt.legend(fontsize='5')
# plt.savefig(graphSavePath  + 'DVH'+'PTV&OAR7B' + str(epoch)+ 'patientid' + str(id1) + 'step' +str(7)+ '.png',dpi=600)
plt.savefig(graphSavePath  + 'DVH'+'PTV&OAR7BwOhelp' + str(epoch)+ 'patientid' + str(id1) + 'step' +str(20)+ '.png',dpi=600)
plt.show()
# plt.show()
plt.close()


        




























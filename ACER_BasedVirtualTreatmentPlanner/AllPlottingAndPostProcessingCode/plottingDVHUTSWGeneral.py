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


def Sum(Array1, Array2, ArrayOAR):
    ArrayOAR = np.array(ArrayOAR)
    result = Array1 + Array2
    # ArraySum = np.zeros((Array.shape[1],))

    for i in range(ArrayOAR.shape[0]):
        result = result + ArrayOAR[i]

    return result

def MinimizeDoseOAR_dvh(MPTV, MOARs, tPTV, tOARs, lambdaPTV, lambdaOARs, VPTV, VOARs, xVec,gamma,pdose,maxiter):
    # treatment planning optimization in DVH-based scheme
    beta=2

    lambdaOARs = np.array(lambdaOARs)
    lambdaOARs = lambdaOARs/lambdaPTV
    # # xVec = np.ones((MPTV.shape[1],))
    DPTV = MPTV.dot(xVec)
    # print('First DPTV Index', DPTV[1])
    DPTV1 = np.sort(DPTV)
    # print('First Sorted DPTV Index', DPTV1[1])
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

        # posi = int(round(0.05 * DPTV1.shape[0]))-1
        # D95 = DPTV1[posi]
        # MPTV95 = MPTV[DPTV >= D95, :]
        # DPTV95 = MPTV95.dot(xVec)

        bs = []
        temps = []
        tempps = np.zeros((len(MOARs),xVec.shape[0]))
        # print('xVec Shape',tempps.shape)
        tempOARs = []
        MOAR1s = []
        MOAR1Ts = []
        # print('length',len(MOARs), len(tOARs))

        for i in range(len(MOARs)):

            DOAR = MOARs[i].dot(xVec)
            DOAR1 = np.sort(DOAR)
            # print('1st DOAR Index', DOAR[1])
            # print('1st sorted DOAR Index', DOAR1[1])
            posi = int(round((1 - VOARs[i]) * DOAR1.shape[0]))-1
            if posi < 0:
                posi = 0
            DOARV = DOAR1[posi]

            MOARV = MOARs[i][DOAR>=DOARV,:]
            temp = DOAR[DOAR>=DOARV]
            # print('First temp Index', temp[1])
            # print('max First temp Index', np.max(temp))
            # print('the i', i, 'running')
            if np.max(temp) > pdose * tOARs[i]:
                # print('the i', i, 'still running')
                MOAR1 = MOARV[temp > pdose * tOARs[i], :]
                targetOAR1 = pdose*tOARs[i]*np.ones((MOAR1.shape[0],))
                # print('first targetOARs', targetOAR1[1])
                MOAR1T = MOAR1.transpose()
                temp3 = MOAR1.dot(xVec)
                # print('temp after Mult xVec 1st', temp3[1])
                temp3 = MOAR1T.dot(temp3)
                # print('temp after Mult xVec 2nd', temp3[1])
                temp3 = temp3 * lambdaOARs[i]/MOAR1.shape[0]
                # print('temp after Mult xVec 3rd', temp3[1])
                if i ==0:
                    b = lambdaOARs[i] * MOAR1T.dot(targetOAR1) / max(MOAR1.shape[0], 1)
                else:   
                    b = lambdaOARs[i] * MOAR1T.dot(targetOAR1) / MOAR1.shape[0]
        
            else:
                temp3 = np.zeros((xVec.shape))
                b = np.zeros((xVec.shape))
                tempps[i,:] = np.zeros((xVec.shape))
            tempOAR = temp
            # print('b sum', np.sum(b))
            # print('temp after Mult xVec final', np.sum(temp3))

            bs.append(b)
            temps.append(temp3)
            tempOARs.append(tempOAR)
            MOAR1s.append(MOAR1)
            MOAR1Ts.append(MOAR1T)

        # templhs = temp1+temp2+temp3+temp4+temp5+temp6+temp7+temp8+temp9
        # b = b1+b2+b3+b4+b5+b6+b7+b8+b9-MPTVT.dot(gamma)
        # templhs = temp1 + temp2 + np.sum(temps, axis = 0)
        templhs = Sum(temp1, temp2, temps)
        # temp1 + temp2 + temps[0]+ temps[1]
        # print('templhs', np.sum(templhs))
        b = Sum(b1, b2, bs)-MPTVT.dot(gamma)
        # print('total b', np.sum(b))
        # if iter == 0:
            # print(temps[0].dtype)
            # print('type', np.sum(temps, axis = 0).shape, temps[0].shape, temps[1].shape)
            # np.save('/data4/tempSum', np.sum(temps, axis = 0))
            # np.save('/data4/temp2', temps[1])
            # np.save('/data4/temp1', temps[0])
            # for k in range(len(temps)):
            #     np.save(f'/data4/temps{k}', temps[k])
            # np.save('/data4/b.npz', b)
            # np.save('/data4/templhs', temp1 +temp2 + np.sum(np.array(temps),axis = 0))
            # np.save('/data4/templhs', templhs)

        r = b - templhs
        # print('total r', np.sum(r))
        p = r
        rsold = np.inner(r,r)

        # print('iter', iter,  rsold)
        # print('p shape', np.sum(p))
        # print()

        # print("rsold=", rsold, "iter=", iter, "=========================")  # this not

        if rsold>1e-10:
            for i in range(3):
            # for i in range(5):
                if np.max(tempptv) > pdose*tPTV :
                    tempp1 = MPTV1.dot(p)
                    tempp1 = MPTV1T.dot(tempp1)
                    tempp1 = tempp1 * 1 / MPTV1.shape[0]


                tempp2 = MPTV.dot(p)
                tempp2 = beta * MPTVT.dot(tempp2)/y.shape[0]

                for j in range(len(tempOARs)):
                    # print(np.max(tempOARs))
                    if np.max(tempOARs[j]) > pdose * tOARs[j]:
                        tempp = MOAR1s[j].dot(p)
                        tempp = MOAR1Ts[j].dot(tempp)
                        tempp = tempp * lambdaOARs[j] / MOAR1s[j].shape[0]

                        # print('tempp i', i, 'j', j, np.sum(tempp))
                        tempps[j,:] = tempp

                # Ap = tempp1 + tempp2 + tempp3 + tempp4 + tempp5 + tempp6 + tempp7 + tempp8 + tempp9
                Ap = Sum(tempp1, tempp2, tempps)
                # print('Ap',i ,  np.sum(Ap))
                pAp = np.inner(p, Ap)
                alpha = rsold / pAp
                xVec = xVec + alpha * p
                # print('xVec final', np.sum(xVec))
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
    # print('OriginalD95', D95)
    factor = pdose / D95 # thresholidng
    xVec = xVec * factor
    converge = 1
    if iter == maxiter - 1:
        converge = 0
    # print("LOOKED HERE DAMON:",converge,iter)
    return xVec, iter


def runOpt_dvh(MPTV, MOARs, tPTV, tOARs, lambdaPTV, lambdaOARs, VPTV, VOARs, xVec,gamma,pdose,maxiter):

    # run optimization and generate DVH curves
    # xVec, iter = MinimizeDoseOAR_dvh(MPTV, MBLA, MREC, MBODY, MFemoralHeadL, MFemoralHeadR, MGenitalia, MPenileBulb,  tPTV, tBLA, tREC, tBODY, tFemoralHeadL, tFemoralHeadR, tGenitalia, tPenileBulb, lambdaPTV, lambdaBLA, lambdaREC, lambdaBODY, lambdaFemoralHeadL, lambdaFemoralHeadR, lambdaGenitalia, lambdaPenileBulb, VPTV, VBLA, VREC, VBODY, VFemoralHeadL, VFemoralHeadR, VGenitalia, VPenileBulb, xVec,gamma,pdose,maxiter)
    xVec, iter = MinimizeDoseOAR_dvh(MPTV, MOARs, tPTV, tOARs, lambdaPTV, lambdaOARs, VPTV, VOARs, xVec,gamma,pdose,maxiter)

#    j = 0
    DPTV = MPTV.dot(xVec)
    DPTV = np.sort(DPTV)
    DPTV = np.flipud(DPTV)

    DOARs = []
    for i in range(len(MOARs)):
        DOAR = MOARs[i].dot(xVec)
        DOAR = np.sort(DOAR)
        DOAR = np.flipud(DOAR)        
        DOARs.append(DOAR)


    INPUT_SIZE = 100
    # ## Plot DVH curve for optimized plan
    edge_ptv = np.zeros((INPUT_SIZE+1,))
    edge_ptv[1:INPUT_SIZE+1] = np.linspace(pdose,pdose*1.15, INPUT_SIZE)
    edge_full = np.zeros((INPUT_SIZE+1,))
    edge_full[1: INPUT_SIZE+1] =  np.linspace(0, pdose*1.2, INPUT_SIZE)
    # edge_ptv[1:INPUT_SIZE + 1] = np.linspace(0, max(DPTV), INPUT_SIZE)
    # x_ptv = np.linspace(pdose+ 0.5* pdose*1.15/INPUT_SIZE,pdose*1.15,INPUT_SIZE)
    x_ptv = np.linspace(0.5 * max(DPTV) / INPUT_SIZE, max(DPTV), INPUT_SIZE)
    (n_ptv, b) = np.histogram(DPTV, bins=edge_ptv)
    (n_PTVFull, bFull) = np.histogram(DPTV, bins = edge_full)
    y_ptv = 1 - np.cumsum(n_ptv / len(DPTV), axis=0)
    y_ptvFull = 1 - np.cumsum(n_PTVFull / len(DPTV), axis=0) 

    # edge_ptvfull = np.zeros((INPUT_SIZE+1,))
    # edge_ptvfull[1:INPUT_SIZE + 1] = np.linspace(0, max(DPTV), INPUT_SIZE)
    # (n_ptv, b) = np.histogram(DPTV, bins=edge_ptvfull)
    # maxDose = max(max(DPTV), max(DBLA),  max(DREC), max(DFemoralHeadL), max(DFemoralHeadR))
 
    Y = np.zeros((INPUT_SIZE, len(tOARs)+1))
    YFull = np.zeros((INPUT_SIZE, len(tOARs)+1))

    Y[:,0] = y_ptv
    YFull[:,0] = y_ptvFull

    edge_oar = np.zeros((INPUT_SIZE + 1,))
    edge_oar[1:INPUT_SIZE + 1] = np.linspace(0.6 * pdose, 1.1 * pdose, INPUT_SIZE)

    for i in range(len(tOARs)):
        # edge_bladder[1:INPUT_SIZE + 1] = np.linspace(0.6 * pdose, maxDose, INPUT_SIZE)
        (n_oar, b) = np.histogram(DOARs[i], bins=edge_oar)
        (n_oarFull, b) = np.histogram(DOARs[i], bins=edge_full)
        Y[:,i+1] = 1 - np.cumsum(n_oar / len(DOARs[i]), axis=0)
        YFull[:,i+1] = 1 - np.cumsum(n_oarFull / len(DOARs[i]), axis=0)
    # J = 

    # plt.plot(edge_ptv[1:101], y_ptv, color = 'red' )
    # plt.plot(edge_oar[1:101], Y[:,1], color = 'green')
    # plt.plot(edge_oar[1:101], Y[:,2], color = 'blue')
    # plt.plot(edge_oar[1:101], Y[:,3], color = 'purple')
    # plt.plot(edge_oar[1:101], Y[:,4], color = 'yellow')
    # plt.show()
    # plt.close()

    Y = np.reshape(Y, (Y.shape[0] * Y.shape[1],), order='F')

    return Y, iter, xVec, YFull

patient_list = ['001', '008', '009', '010', '011', '013', '014', '015', '016', '017', '018', '020', '022', '023', '025', '026', '027', '028', '030', '031', '036', '037', '039', '042', '043', '045', '046', '054', '057', '061', '065', '066', '068', '070', '073', '074', '077', '080', '081', '083', '084', '085', '087', '088', '091', '093', '095', '097', '098']
#patient_list = [ '008',  '010',     '020',    '027',   '031',   '039', '042',  '046',  '061',  '070',    '084',  '087',  '092',  '095',  '098']


# data_result_path = '/data2/mainul/results_CORS1Paper/scratch6_30StepsNewParamenters3NewData1/dataWithPlanscoreRun/'
# data_result_path = '/data2/mainul/results_CORS1PaperDiffBeam/scratch6_30StepsNewParamenters3NewData1/dataWithPlanscoreRun/'
data_result_path = '/data2/mainul/results_CORS1PaperDiffBeam7/scratch6_30StepsNewParamenters3NewData1/dataWithPlanscoreRun/'
# data_result_path = '/data2/mainul/results_CORS1random6C/scratch6_30StepsNewParamenters3NewData1/dataWithPlanscoreRun/'
# graphSavePath = '/data2/mainul/DataAndGraph/'
# data_result_path = '/data4/mainul/MultiModalAI6Beam/Blafixed/PaperInitLSTM155500/dataWithPlanscoreRun/'
data_result_path = '/data4/mainul/MultiModalAI6Beam/Blafixed/PaperInitLSTM155500/dataWithPlanscoreRun/'
# data_result_path = '/data4/mainul/MultiModalAI/Blafixed/PaperInitLSTM155500/dataWithPlanscoreRun/'
data_result_path = '/data4/mainul/MultiModalAIVMAT/PaperInitLSTM/dataWithPlanscoreRun/'

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

# from lib_dvh.data_prep import loadDoseMatrix
# from lib_dvh.data_prep_diff_beam import loadDoseMatrix, loadMask, ProcessDmat
from lib_dvh.data_prepVMATallOrgans import loadDoseMatrix,loadMask,ProcessDmat
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

OARs = ['bladder', 'rectum', 'fem_head_lt', 'fem_head_rt']

targetLabels, PTVLabel, OARlabels = loadMask(data_path2+str(id1)+'.h5', OARs)
doseMatrix = loadDoseMatrix(data_path+str(id1)+'.hdf5')

MPTV, _, MOARs = ProcessDmat(doseMatrix, targetLabels, OARlabels)

# NpzFile = np.load(data_result_path+str(id1)+'tpptuning120499.npz')
# NumberOfSteps = len(np.nonzero(NpzFile['l1'])[0])
# print(NumberOfSteps)
NpzFile = [1.0, 0.6, 0.6, 0.6, 0.6, 54.93783665, 1.0, 0.6, 0.6, 0.6, 0.1, 1.0, 1.0, 1.0, 1.0]


# tPTV = NpzFile['l1'][2]
# tBLA = NpzFile['l2'][2]
# tREC = NpzFile['l3'][2]
# lambdaPTV = NpzFile['l4'][2]
# lambdaBLA = NpzFile['l5'][2]
# lambdaREC = NpzFile['l6'][2]
# VPTV = NpzFile['l7'][2]
# VBLA = NpzFile['l8'][2]
# VREC = NpzFile['l9'][2]

tPTV = 1.0
tOARs = [0.6, 0.6, 0.6, 0.6]
lambdaPTV = 54.93783665
lambdaOARs = [1.0, 0.6, 0.6, 0.6]
VPTV = 0.1
VOARs = [1.0,1.0,1.0,1.0,]
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
Y, iter, xVec, Yfull = runOpt_dvh(MPTV, MOARs, tPTV, tOARs, lambdaPTV, lambdaOARs, VPTV, VOARs, xVec, gamma, pdose, maxiter)



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
colors = ['red', 'green', 'blue', 'yellow', 'purple']
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

for i in range(5):
    plt.plot(edge_Full[1:INPUT_SIZE + 1]*100, Yfull[:, i]*100, color = colors[i], linestyle = '-')


DoseBladder = [(65/79.5)*100, (70/79.5)*100, (75/79.5)*100, (80/79.5)*100]
DoseRectum = [(60/79.5)*100, (65/79.5)*100, (70/79.5)*100, (75/79.5)*100]
Dosefmhead = [37, 50, 55]
DosePTV = [100, 110]
Volume = [ 55, 40, 30, 20]
VolumePTV = [95, 1.0]
VolumefmHead = [50, 35, 5]

# DoseBlaToPlot = [(DoseBladder[i]/120)*100 for i in range(4)

plt.scatter(DoseBladder, Volume, marker = 's', color ='black', s = 200)
plt.scatter(DoseRectum, Volume, marker = '^', color ='black', s = 200)
plt.scatter(DosePTV, VolumePTV, marker = '*', color = 'black', s = 200)
plt.scatter(Dosefmhead, VolumefmHead, marker = 'o', color ='black', s = 200)

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
plt.savefig(graphSavePath  + 'DVH'+'PTV&4OAR' + str(epoch)+ 'patientid' + str(id1) + 'step' +str(2)+ '.png',dpi=600)
plt.show()
# plt.show()
plt.close()


        




























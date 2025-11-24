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

sys.path.append("/home/mainul/Actor-critic-based-treatment-planning/acer_VTPN-12-18-23/")
# for i in range(10):
#     # some code to generate data
#     data = np.random.rand(10)  # huge array of data points
#     print(f'data:{data}')
#     np.save(f'data_{i}.npy', data)  # save data to disk as binary file
#
# print("Saving finished")
# data_points = []
# for i in range(10):
#     data = np.load(f'data_{i}.npy')
#     data_points.append(data)
#     print(f'data_points:{data_points}')
# plt.plot(np.concatenate(data_points))
# plt.show()

patient_list = ['001', '008', '009', '010', '011', '013', '014', '015', '016', '017', '018', '020', '022', '023', '025', '026', '027', '028', '030', '031', '036', '037', '039', '042', '043', '045', '046', '054', '057', '061', '065', '066', '068', '070', '073', '074', '077', '080', '081', '083', '084', '085', '087', '088', '091', '093', '095', '097', '098']

# data_result_path = '/data2/mainul/results_CORS1Paper/scratch6_30StepsNewParamenters3NewData1/dataWithPlanscoreRun/'
data_result_path = '/data2/mainul/results_CORS1random6C/scratch6_30StepsNewParamenters3NewData1/dataWithPlanscoreRun/'
graphSavePath = '/data2/mainul/DataAndGraph/'


def maximum_step(patientid):
    # data_path = '/data/mainul1/results_CORS1/scratch6_30StepsNewParamenters3NewData4/dataWithPlanscoreRun/'
    # data_path = '/data2/mainul/results_CORS1Paper/scratch6_30StepsNewParamenters3NewData1/dataWithPlanscoreRun/'
    data_path = '/data2/mainul/results_CORS1random6C/scratch6_30StepsNewParamenters3NewData1/dataWithPlanscoreRun/'
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

from lib_dvh.data_prep import loadDoseMatrix
# id1 = 0
# The next line is enough for random
id1 = 67
# Also add the following line for paper data
# id1 = patient_list[id1]

# doseMatrix = np.load(data_result_path+str(id1)+'doseMatrix' + str(epoch)  + 'step' + str(0)+'.npy',allow_pickle = True)
# doseMatrix = loadDoseMatrix('/home/mainul1/Actor-critic-based-treatment-planning/acer_VTPN-12-18-23/test_onceagain.hdf5')
# doseMatrix = loadDoseMatrix('/home/mainul1/Actor-critic-based-treatment-planning/acer_VTPN-12-18-23/test_TORTS.hdf5')
# doseMatrix = loadDoseMatrix('/media/mainul/Chi-Drives/128641-3-E21/var/lib/docker/overlay2/8fa0092f3bc6971c1752f734f8fd34c782377f383ffe6f5a55629a01d3b0f185/diff/home/exx/dose_deposition_full/prostate_dijs/f_dijs/008.hdf5')
# print(doseMatrix.shape)
#==================
bladderLabel = np.load(data_result_path+str(id1-1)+'bladderLabel' + str(epoch)  + 'step' + str(0)+'.npy',allow_pickle = True)
rectumLabel= np.load(data_result_path+str(id1-1)+'rectumLabel' + str(epoch)  + 'step' + str(0)+'.npy',allow_pickle = True)
PTVLabel = np.load(data_result_path+str(id1-1)+'PTVLabel' + str(epoch)  + 'step' + str(0)+'.npy',allow_pickle = True)


structureFilePath = '/media/mainul/Chi-Drives/128641-3-E21/var/lib/docker/overlay2/8fa0092f3bc6971c1752f734f8fd34c782377f383ffe6f5a55629a01d3b0f185/diff/home/exx/dose_deposition_full/plostate_dijs/f_masks/'

print(bladderLabel.shape)
print(rectumLabel.shape)
print(PTVLabel.shape)
#======================================================
structureFile = structureFilePath + str(patient_list[int((id1-1)//3)])+ '.h5'
# The following line is for paper dataset====================================
# structureFile = np.load(structureFilePath + str(id1)+ '.h5')
# #=========================================
# structureFile = '/media/mainul/Chi-Drives/128641-3-E21/var/lib/docker/overlay2/8fa0092f3bc6971c1752f734f8fd34c782377f383ffe6f5a55629a01d3b0f185/diff/home/exx/dose_deposition_full/plostate_dijs/f_masks/008.h5'
# _, bladderLabel1, rectumLabel1, PTVLabel1 = loadMask('/media/mainul/Chi-Drives/128641-3-E21/var/lib/docker/overlay2/8fa0092f3bc6971c1752f734f8fd34c782377f383ffe6f5a55629a01d3b0f185/diff/home/exx/dose_deposition_full/plostate_dijs/f_masks/008.h5')

# print(bladderLabel1.shape)
# print(rectumLabel1.shape)
# print(PTVLabel1.shape)
# #===================================================
# print(max(doseMatrix.indices))



mask = h5py.File(structureFile,'r')
dosemask = mask['oar_ptvs']['dose']
dosemask1 = dosemask
dosemask = np.reshape(dosemask, (dosemask.shape[0] * dosemask.shape[1] * dosemask.shape[2],), order='C')
dosemaskNonzero = np.nonzero(dosemask)[0]
NewDose = np.zeros(dosemask.shape[0])
NewPTVlabel = np.zeros(dosemask.shape[0])
NewBladderLabel = np.zeros(dosemask.shape[0])
NewRectumLabel = np.zeros(dosemask.shape[0])
# Dose = np.load("/data2/mainul/results_CORS1random6C/scratch6_30StepsNewParamenters3NewData1/dataWithPlanscoreRun/4Dose120499step22.npy")

# print(Dose.shape)

print('first nonzero', dosemaskNonzero.shape)

# print("old indices shape", doseMatrix.indices.shape)

# NewIndices = doseMatrix.indices
 
# for i in range(Dose.shape[0]):
#     NewDose[dosemaskNonzero[i]] = Dose[i]
#     # if i%500==0:
    #     print(i)

print(NewDose.shape)
print(np.array(np.nonzero(NewDose)).shape)

for i in range(PTVLabel.shape[0]):
    NewPTVlabel[dosemaskNonzero[i]] = PTVLabel[i]
    # if i%500==0:
    #     print(i)

print(NewPTVlabel.shape)
print(np.array(np.nonzero(NewPTVlabel)).shape)

for i in range(bladderLabel.shape[0]):
    NewBladderLabel[dosemaskNonzero[i]] = bladderLabel[i]
    # if i%500==0:
    #     print(i)

print(NewBladderLabel.shape)
print(np.array(np.nonzero(NewBladderLabel)).shape)

for i in range(rectumLabel.shape[0]):
    NewRectumLabel[dosemaskNonzero[i]] = rectumLabel[i]
    # if i%500==0:
    #     print(i)

print(NewRectumLabel.shape)
print(np.array(np.nonzero(NewRectumLabel)).shape)

# xVec = np.ones((doseMatrix.shape[1]))

# Dose = doseMatrix.dot(xVec)

# plt.imshow(dosemask1[100], cmap = 'jet', interpolation ='nearest')
# plt.colorbar()
# plt.show()
# NewData = doseMatrix.data
# newColumIndex = np.nonzero(dosemask)[0]
# Newindptr = doseMatrix.indptr
# print(NewData.shape)
# print(dosemask.shape[0])
# FullDij =  csc_matrix((NewData, newColumIndex, Newindptr), shape=(dosemask.shape[0], doseMatrix.shape[1]))
# print(FullDij.shape)

# doseMatrix = doseMatrix.tocsr()

# for i in range(doseMatrix.shape[0]):
#     FullDij[np.nonzero(dosemask)[0][i],:] = doseMatrix.getrow(i).toarray()
#     if i%500 == 0:
#         print(i)

# np.save('/data2/mainul/DoseFull', FullDij)
# print('Full Shape', FullDij.shape)

# print('nonzero shape', np.nonzero(FullDij).shape) 


# ## Next 3 lines are for CORT
# # depth = 90 
# # rows = 184
# # cols = 184

# Next three lines are for for TROTS(2nd patient)
# depth = 242
# rows = 264
# cols = 411
# 162, 104, 143
PTVLabel = NewPTVlabel
bladderLabel = NewBladderLabel
rectumLabel = NewRectumLabel



# depth = 104
# rows = 143
# cols = 162
# depth = 162
# rows = 143
# cols = 104
depth = dosemask1.shape[0]
rows = dosemask1.shape[2]
cols = dosemask1.shape[1]
# depth = 162
# rows = 143
# cols = 104
# rows= 162
# cols = 104
print(143*104*162)

final_mask = []
total_voxel = depth*rows*cols
voxel_vector = np.arange(1, total_voxel + 1)


# dosemasktemp = np.unique(doseMatrix.indices)
dosemasktemp = dosemaskNonzero
idx1 = [x + 1 for x in dosemasktemp]
final_mask = np.union1d(final_mask, idx1)
dosemask = np.isin(voxel_vector, final_mask).astype(int)

dm_3d = dosemask.reshape((depth, cols, rows))
# dm_3d = np.transpose(dm_3d, (0,2,1))
dm_3d = np.transpose(dm_3d, (1, 0, 2))
# (1, 0, 2)

print("np.nonzero(PTVLabel)",np.nonzero(PTVLabel))
pl_3d = PTVLabel.reshape((depth, cols, rows))
pl_3d = np.transpose(pl_3d, (1, 0, 2))
# pl_3d = np.transpose(pl_3d, (0,2,1))

print("np.nonzero(bladderLabel)",np.nonzero(bladderLabel))
bl_3d = bladderLabel.reshape((depth, cols, rows))
bl_3d = np.transpose(bl_3d, (1, 0, 2))
# bl_3d = np.transpose(bl_3d, (0,2,1))


print("np.nonzero(rectumLabel)",np.nonzero(rectumLabel))
rl_3d = rectumLabel.reshape((depth, cols, rows))
rl_3d = np.transpose(rl_3d, (1, 0, 2))
# rl_3d = np.transpose(rl_3d, (0,2,1))



# # Plotting the DVH graphs
# print("Plotting the DVH graphs")
# data_result_path2 = './data/data/Results/figuresPATp/CORS/3-2023-11-25/non-G/DVH_graphs/'
# for patientid in range(evaluation_episodes):
    # print("Patient_ID",patientid)
    # for episode_length in range(max_episode_length):
        # try:
            # Y = np.load(data_result_path+str(patientid)+'xDVHY' + str(epoch)  + 'step' + str(episode_length)+'.npy')
        # except Exception as e:
            # print(e)
        # plt.plot(Y[:, 3], Y[:, 0])
        # plt.plot(Y[:, 4], Y[:, 1])
        # plt.plot(Y[:, 5], Y[:, 2])
        # plt.legend(('ptv', 'bladder', 'rectum'))
        # plt.title(str(patientid)+ 'DVH' + str(epoch) + 'step' + str(episode_length))
        # plt.savefig(data_result_path2 + str(patientid) + 'DVH' + str(epoch) + 'step' + str(episode_length) + '.png')
        # plt.close()
        # print(f'Figure{episode_length} done')


# # Plotting the tpptuning
# data_result_path2 = './data/data/Results/figuresPATp/CORS/3-2023-11-25/non-G/TPP_tuning/'
# print("Plotting the tpptuning")
# for patientid in range(evaluation_episodes):
    # name1 = data_result_path + str(patientid) + 'tpptuning' + str(epoch)
    # tpp_parameters= np.load(name1+'.npz')
    # plt.plot(tpp_parameters['l1'])
    # plt.plot(tpp_parameters['l2'])
    # plt.plot(tpp_parameters['l3'])
    # plt.plot(tpp_parameters['l4'])
    # plt.plot(tpp_parameters['l5'])
    # plt.plot(tpp_parameters['l6'])
    # plt.plot(tpp_parameters['l7'])
    # plt.plot(tpp_parameters['l8'])
    # plt.plot(tpp_parameters['l9'])

    # plt.legend(('tPTV', 'tBLA', 'tREC', 'lambdaPTV', 'lambdaBLA', 'lambdaREC', 'VPTV', 'VBLA', 'VREC'))
    # plt.title(str(patientid)+'TPP tuning steps')
    # plt.savefig(data_result_path2 + str(patientid)+'TPPtuning'+str(epoch) + '.png')
    # plt.close()
    # print(f'Figure{patientid} done')
    
    
 
# #Plotting the heatmap
# data_result_path2 = './data/data/Results/figuresPATp/CORS/3-2023-11-25/non-G/Heatmap_test/'
# Next line is for CORT
# patient_num = [1]
# #Next line is for TORTS
patient_num = [id1-1]
for patientid in patient_num:
    print("Plotting the Heatmap")
    maxStep = maximum_step(patientid)
    step_num = [0, 10, maxStep]
    # step_num = [0]

    # # Next line is for CORT 
    # for slice_num in range(39,51):
    for slice_num in range(55,56):
    # for slice_num in step_num:
    # for slice_num in step_num:
        # patientid = 14
        print("slice_num",slice_num)
        dm_2d = dm_3d[:,:, slice_num]
        pl_2d = pl_3d[:,:, slice_num]
        bl_2d = bl_3d[:,:, slice_num]
        rl_2d = rl_3d[:,:, slice_num]
        
        
        # if not os.path.exists(f"/data/mainul1/results_CORS1/scratch6_30StepsNewParamenters3NewData4/colorwash/{slice_num}/"): 
        #     os.makedirs(f"/data/mainul1/results_CORS1/scratch6_30StepsNewParamenters3NewData4/colorwash/{slice_num}/")
        # data_result_path2 = f"/data/mainul1/results_CORS1/scratch6_30StepsNewParamenters3NewData4/colorwash/{slice_num}/"
        for episode_length in step_num:
            try:
                Dose = np.load(data_result_path+str(patientid)+'Dose' + str(epoch)  + 'step' + str(episode_length)+'.npy',allow_pickle = True)
                NewDose = np.zeros(dosemask.shape[0])
                for i in range(Dose.shape[0]):
                    NewDose[dosemaskNonzero[i]] = Dose[i]
                    # if i%500==0:
                    #     print(i)

                print(NewDose.shape)
                print(np.array(np.nonzero(NewDose)).shape)

                Dose = NewDose
                Dose = Dose.reshape((depth, cols, rows))
                # Dose = np.transpose(Dose, (0,2,1))
                Dose = np.transpose(Dose, (1, 0, 2))
                
                Dose_2d = Dose[:,:, slice_num]
                colors = [
                    (0, "blue"),
                    (0.20, "cyan"),
                    (0.45, "yellow"),
                    (0.60, "orange"),
                    (0.75, "red"),
                    (0.85, "firebrick"),
                    (0.95, "maroon"),
                    (1.0, "darkred"),
                ]


                # Create a custom colormap
                cmap = mcolors.LinearSegmentedColormap.from_list("BlueBrown", colors)
                vmax = 1.3
                vmin = 0.0

                # plt.imshow(Dose_2d, cmap = 'jet', interpolation ='nearest')
                # plt.imshow(Dose_2d, cmap = 'jet', interpolation ='nearest')
                plt.imshow(Dose_2d, vmin = vmin, vmax = vmax, cmap = cmap, interpolation ='nearest')
                plt.colorbar()
                plt.contour(dm_2d, colors = 'black')
                # Next three lines need to be commented for TORTS
                plt.contour(pl_2d, colors = 'black')
                plt.contour(bl_2d,colors = 'green')
                plt.contour(rl_2d,colors = 'blue')
                
                plt.savefig(graphSavePath + str(patientid)+'Heatmap'+ str(epoch) + 'step' + str(episode_length) +'.png', dpi = 1200)
                # plt.show()
                plt.close()
                print(f'Figure{episode_length} done')
            except Exception as e:
                print(e) 
        



























